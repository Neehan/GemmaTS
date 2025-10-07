"""Quick demo configuration for Chronos Bolt."""

from dataclasses import dataclass
from src.configs.base.etth1 import ETTh1BaseConfig


@dataclass
class Config(ETTh1BaseConfig):
    """Quick demo configuration for Chronos Bolt baseline.

    Inherits common ETTh1 settings, only defines Chronos-specific parameters.
    """

    # Chronos model
    chronos_pretrained: str = "amazon/chronos-bolt-tiny"

    # Chronos architecture
    input_patch_size: int = 16
    input_patch_stride: int = 8
    freeze: bool = True
    num_train_epochs: int = 5

    # Output path
    output_dir: str = "data/checkpoints/chronos_bolt_etth1_demo"
