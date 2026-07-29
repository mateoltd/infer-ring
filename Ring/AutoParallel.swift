import MLX
import MLXLMCommon

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
    print(
        "Warning: pipeline sharding is unavailable for \(type(of: model)); "
            + "using the unmodified model"
    )
    return model
}
