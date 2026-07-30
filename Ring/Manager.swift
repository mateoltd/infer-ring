//
import Foundation
import Darwin
import MLX
import MLXLMCommon
import MLXLLM
import MLXVLM
import MLXNN

private struct ModelCacheDownloader: Downloader {
    let directory: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        directory
    }
}

public final class MLXManager {
    public init() {}

    private var group: DistributedGroup?
    private var groupConfiguration: GroupConfiguration?

    private struct GroupConfiguration: Equatable {
        let rank: Int
        let devices: [String]
    }

    public func initMLX(rank: Int, devices: [String]) throws {
        let requestedConfiguration = GroupConfiguration(rank: rank, devices: devices)
        if let groupConfiguration {
            guard groupConfiguration == requestedConfiguration else {
                throw RingError.failed(
                    "MLX distributed group is already initialized with a different topology; restart Infer Ring before reforming the ring"
                )
            }
            print("MLX ring already initialized; reusing rank \(rank)")
            return
        }

        let port = 13373
        let json = try JSONEncoder().encode(devices.map {
            ["\($0):\(port)", "\($0):\(port+1)"]
        })
        let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        if !FileManager.default.fileExists(atPath: cachesDir.path) {
            try FileManager.default.createDirectory(at: cachesDir, withIntermediateDirectories: true)
        }
        let hostfileUrl = cachesDir.appendingPathComponent("mlx_hostfile.json")
        try json.write(to: hostfileUrl)
        print("Initializing MLX ring with rank \(rank)")
        setenv("MLX_HOSTFILE", hostfileUrl.path, 1)
        setenv("MLX_RANK", "\(rank)", 1)
        #if DEBUG
        setenv("MLX_RING_VERBOSE", "1", 1)
        #endif

        try MLX.withError {
            group = DistributedGroup.initialize(strict: true)
        }
        groupConfiguration = requestedConfiguration
    }

    public func synchronize() {
        guard let group else {
            print("group not initialized")
            return
        }

        group.allSum(MLXArray(1.0)).eval()
    }

    public func validate() {
        guard let group else {
            print("group not initialized")
            return
        }
        let key = MLXRandom.key(0)
        let value = MLXRandom.uniform(-100.0 ..< 100, [2,4,6], key: key)
        let f16 = value.asType(.float16)
        let sum = group.allSum(f16)
        let size = Float(group.size)
        let expected = f16 * size
        let diff = abs(sum - expected).max()
        
        if diff.item(Float.self) < 1e-3 {
            print("Distributed validation passed!")
        } else {
            print("Distributed validation failed! Max difference: \(diff.item(Float.self))")
        }
    }

    public func loadModel(
        _ card: ModelCard,
        shardMeta: ShardMetadata,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        Memory.clearCache()

        let configuration = ModelConfiguration(
            id: card.modelId,
            eagerlyEvaluateWeights: group == nil
        )
        let downloader = ModelCacheDownloader(directory: card.cacheDirectory)
        let tokenizerLoader = SwiftTransformersTokenizerLoader()
        var context: ModelContext
        if card.isVisionModel {
            context = try await VLMModelFactory.shared.load(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration,
                progressHandler: progressHandler
            )
        } else {
            context = try await LLMModelFactory.shared.load(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration,
                progressHandler: progressHandler
            )
        }

        if let group {
            if shardMeta.useTensorParallel && card.metadata.supportsTensor {
                context.model = tensorAutoParallel(
                    model: context.model,
                    group: group
                )
            }
            else {
                context.model = pipelineAutoParallel(
                    model: context.model,
                    group: group,
                    modelShardMeta: shardMeta
                )
            }
            eval(context.model)
        }

        return context
    }

    /// Load the weighted Qwen MTP sidecar only on rank 0. Pipeline workers
    /// use a weightless follower that receives rank 0's proposal block before
    /// all ranks enter target verification.
    public func loadQwenMTPDrafter(
        from directory: URL
    ) async throws -> MTPDrafterContainer {
        await Qwen35MTPRegistration.register()

        if let group, Int(group.rank) != 0 {
            let context = MTPDrafterContext(
                configuration: ModelConfiguration(
                    id: "mlx-community/Qwen3.6-35B-A3B-MTP-4bit-follower"
                ),
                model: Qwen35MTPFollowerModel()
            )
            return MTPDrafterContainer(context: context)
        }

        let tokenizerLoader = SwiftTransformersTokenizerLoader()
        let context = try await MTPDrafterModelFactory.shared.load(
            from: directory,
            using: tokenizerLoader
        )
        eval(context.model)
        return MTPDrafterContainer(context: context)
    }

}
