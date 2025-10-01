"""Pure Chronos Bolt model for baseline comparison."""

from chronos.chronos_bolt import ChronosBoltModelForForecasting


def create_chronos_bolt(
    chronos_base: str,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    patch_stride: int,
):
    """Create pure Chronos Bolt model from pretrained weights."""
    model = ChronosBoltModelForForecasting.from_pretrained(chronos_base)

    model.config.chronos_config["context_length"] = context_length
    model.config.chronos_config["prediction_length"] = prediction_length
    model.config.chronos_config["input_patch_size"] = patch_size
    model.config.chronos_config["input_patch_stride"] = patch_stride

    return model
