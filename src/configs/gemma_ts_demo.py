"""Quick demo configuration for GemmaTS."""

from dataclasses import dataclass
from src.configs.gemma_ts import Config as GemmaTSConfig


@dataclass
class Config(GemmaTSConfig):
    """Quick demo configuration for GemmaTS.

    Inherits common ETTh1 settings, uses reduced epochs for quick testing.
    """

    # Training - quick demo with fewer epochs
    num_train_epochs: int = 5

    # Output path
    output_dir: str = "data/checkpoints/gemma_ts_etth1_demo"
