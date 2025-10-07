"""Quick demo configuration for GemmaTS."""

from dataclasses import dataclass
from src.configs.base.etth1 import ETTh1BaseConfig


@dataclass
class Config(ETTh1BaseConfig):
    """Quick demo configuration for GemmaTS.

    Inherits common ETTh1 settings, uses reduced epochs for quick testing.
    """

    # Model paths
    chronos_pretrained: str = "amazon/chronos-bolt-tiny"
    gemma_model_name: str = "google/gemma-3-270m"

    # Architecture
    input_patch_size: int = 16
    input_patch_stride: int = 8
    text_prompt: str = "Predict the next values in this time series:"
    freeze: bool = True

    # Training - quick demo with fewer epochs
    num_train_epochs: int = 5

    # Output path
    output_dir: str = "data/checkpoints/gemma_ts_etth1_demo"
