"""Pure Chronos Bolt model for baseline comparison."""

import torch
from chronos.chronos_bolt import ChronosBoltModelForForecasting


def create_chronos_bolt(
    chronos_base: str,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    patch_stride: int,
    freeze: bool = True,
    use_bfloat16: bool = True,
):
    """Create pure Chronos Bolt model from pretrained weights.

    Note: Chronos Bolt models have a fixed prediction_length determined at training time.
    For amazon/chronos-bolt, this is 64. The output layers are sized accordingly.
    You cannot change prediction_length without retraining the output head.
    """
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = ChronosBoltModelForForecasting.from_pretrained(
        chronos_base, torch_dtype=dtype
    )

    # Verify the prediction_length matches the model's architecture
    model_pred_len = model.chronos_config.prediction_length
    if prediction_length != model_pred_len:
        print(
            f"WARNING: Requested prediction_length={prediction_length} but model supports {model_pred_len}"
        )
        print(f"Using model's native prediction_length={model_pred_len}")
        prediction_length = model_pred_len

    # Only update context_length and patch settings (these don't affect output layer size)
    model.config.chronos_config["context_length"] = context_length
    model.chronos_config.context_length = context_length

    model.config.chronos_config["input_patch_size"] = patch_size
    model.chronos_config.input_patch_size = patch_size

    model.config.chronos_config["input_patch_stride"] = patch_stride
    model.chronos_config.input_patch_stride = patch_stride

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        # output layer params
        for param in model.output_patch_embedding.parameters():
            param.requires_grad = True

    return model
