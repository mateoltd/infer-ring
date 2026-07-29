import MLX
import MLXLMCommon
import MLXLLM

/// Compatibility boundary for pipeline sharding.
///
/// Current MLX Swift keeps model layer implementations internal, so sharding
/// must be implemented by the model itself rather than by replacing layers
/// from this package. Until a model opts into that interface, preserve the
/// fully loaded model instead of applying an unsafe partial transformation.
public func pipelineAutoParallel(
    model: any LanguageModel,
    group: DistributedGroup,
    modelShardMeta: ShardMetadata
) -> any LanguageModel {
    if let qwen = model as? Qwen35TextModel {
        qwen.configurePipeline(
            startLayer: modelShardMeta.startLayer,
            endLayer: modelShardMeta.endLayer,
            rank: modelShardMeta.deviceRank,
            worldSize: modelShardMeta.worldSize,
            group: group
        )
        return model
    }
    if let qwen = model as? Qwen35Model {
        qwen.configurePipeline(
            startLayer: modelShardMeta.startLayer,
            endLayer: modelShardMeta.endLayer,
            rank: modelShardMeta.deviceRank,
            worldSize: modelShardMeta.worldSize,
            group: group
        )
        return model
    }

    print(
        "Warning: pipeline sharding is unavailable for \(type(of: model)); "
            + "using the unmodified model"
    )
    return model
}
