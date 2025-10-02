"""PatchTST configuration."""

from dataclasses import dataclass
from src.configs.base.etth1 import ETTh1BaseConfig


@dataclass
class Config(ETTh1BaseConfig):
    """Configuration for PatchTST baseline.

    Inherits common ETTh1 settings, only defines PatchTST-specific architecture.
    """

    # PatchTST architecture (from paper Table 13 - ETTh1 univariate)
    enc_in: int = 1  # Univariate
    e_layers: int = 3
    n_heads: int = 4
    d_model: int = 16
    d_ff: int = 128
    dropout: float = 0.3
    fc_dropout: float = 0.3
    head_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    individual: bool = False
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    decomposition: bool = False
    kernel_size: int = 25
    padding_patch: str = "end"

    # Output path
    output_dir: str = "data/checkpoints/patchtst_etth1"
