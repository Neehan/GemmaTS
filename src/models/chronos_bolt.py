"""Pure Chronos Bolt model for baseline comparison."""

from chronos.chronos_bolt import ChronosBoltModelForForecasting
from transformers import AutoConfig


def create_chronos_bolt(
    chronos_base: str,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    patch_stride: int,
):
    """Create pure Chronos Bolt model from pretrained config."""
    config = AutoConfig.from_pretrained(chronos_base)

    config.chronos_config["context_length"] = context_length
    config.chronos_config["prediction_length"] = prediction_length
    config.chronos_config["input_patch_size"] = patch_size
    config.chronos_config["input_patch_stride"] = patch_stride

    return ChronosBoltModelForForecasting(config)
