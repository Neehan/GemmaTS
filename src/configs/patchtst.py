"""PatchTST configuration for pred_len=64 (to match Chronos Bolt)."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for PatchTST baseline with pred_len=64."""

    # Data config
    data: str = "ETTh1"
    root_path: str = "data/datasets/ETT-small/"
    data_path: str = "ETTh1.csv"
    features: str = "S"  # Univariate
    target: str = "OT"
    freq: str = "h"
    embed: str = "timeF"

    # Model config
    seq_len: int = 512  # Match Chronos context length
    label_len: int = 0  # Not used in PatchTST
    pred_len: int = 64  # Match Chronos Bolt prediction length

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

    # Training config
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    num_train_epochs: int = 20
    max_steps: int = -1
    eval_steps: int = 500
    save_steps: int = 500
    warmup_steps: int = 100
    logging_steps: int = 50

    # Paths
    output_dir: str = "data/checkpoints/patchtst_etth1_64"
    seed: int = 42
