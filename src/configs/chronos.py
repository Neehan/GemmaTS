"""Chronos Bolt baseline configuration."""

from dataclasses import dataclass
from src.configs.base.etth1 import ETTh1BaseConfig


@dataclass
class Config(ETTh1BaseConfig):
    """Configuration for Chronos Bolt baseline.

    Inherits common ETTh1 settings, only defines Chronos-specific parameters.
    """

    # Chronos model
    chronos_pretrained: str = "amazon/chronos-bolt-tiny"

    # Chronos architecture
    input_patch_size: int = 16
    input_patch_stride: int = 8
    freeze: bool = True

    # Output path
    output_dir: str = "data/checkpoints/chronos_bolt_etth1"
