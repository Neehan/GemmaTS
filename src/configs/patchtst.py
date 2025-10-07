"""PatchTST configuration."""

from dataclasses import dataclass
from src.configs.base.etth1 import ETTh1BaseConfig


@dataclass
class Config(ETTh1BaseConfig):
    """Configuration for PatchTST baseline.

    Inherits common ETTh1 settings, only defines PatchTST-specific architecture.
    """

    # Data config - override for multivariate
    features: str = "S"  # Multivariate (all 7 features)

    # PatchTST architecture (from paper - ETTh1 multivariate)
    enc_in: int = 1  # Multivariate (7 features in ETTh1)
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

    # Loss function
    loss_fn: str = "mse"

    # Output path
    output_dir: str = "data/checkpoints/patchtst_etth1"
