"""Quick demo configuration for Chronos Bolt."""

from dataclasses import dataclass
from src.configs.chronos import Config as ChronosConfig


@dataclass
class Config(ChronosConfig):
    """Quick demo configuration for Chronos Bolt baseline.

    Inherits common ETTh1 settings, only defines Chronos-specific parameters.
    """

    num_train_epochs: int = 5

    # Output path
    output_dir: str = "data/checkpoints/chronos_bolt_etth1_demo"
