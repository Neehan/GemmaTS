"""Base configuration for ETTh1 dataset - common settings across all models."""

from dataclasses import dataclass


@dataclass
class ETTh1BaseConfig:
    """Base configuration for ETTh1 dataset.

    These settings are common across all models (GemmaTS, Chronos, PatchTST).
    Individual model configs inherit from this and override model-specific parameters.
    """

    # Data config - shared by all models
    data: str = "ETTh1"
    root_path: str = "data/datasets/ETT-small/"
    data_path: str = "ETTh1.csv"
    features: str = "S"  # Univariate
    target: str = "OT"
    freq: str = "h"
    embed: str = "timeF"

    # Sequence lengths - maintain 3.5x ratio from PatchTST paper (336/96 = 3.5)
    seq_len: int = 224  # Context length (64 * 3.5 = 224)
    pred_len: int = 64  # Prediction length (matches Chronos Bolt capability)
    label_len: int = 0  # Not used in our models

    # Training config - shared by all models
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    num_train_epochs: int = 20
    max_steps: int = -1
    eval_steps: int = 500
    save_steps: int = 500
    warmup_steps: int = 100
    logging_steps: int = 50
    seed: int = 42
