import MLX
import MLXLMCommon

/// Preserve correctness when a model does not expose a current tensor-sharding interface.
public func tensorAutoParallel(
    model: any LanguageModel,
    group: DistributedGroup
) -> any LanguageModel {
    print(
        "Warning: tensor sharding is unavailable for \(type(of: model)); "
            + "using the unmodified model"
    )
    return model
}
