"""Full training configuration for GemmaTS."""

from dataclasses import dataclass
from src.configs.base.etth1 import ETTh1BaseConfig


@dataclass
class Config(ETTh1BaseConfig):
    """Configuration for full GemmaTS training.

    Inherits common ETTh1 settings, only defines GemmaTS-specific parameters.
    """

    # Model paths
    chronos_pretrained: str = "amazon/chronos-bolt-mini"
    gemma_model_name: str = "google/gemma-3-4b-pt"

    # Architecture
    input_patch_size: int = 16
    input_patch_stride: int = 8
    text_prompt: str = "Predict the next value in this time series: "
    freeze: bool = True

    # Loss function
    loss_fn: str = "mse"

    # Output path
    output_dir: str = "data/checkpoints/gemma_ts_etth1"
