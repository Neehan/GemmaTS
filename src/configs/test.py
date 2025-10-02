"""Test configuration for quick verification."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for GemmaTS testing."""

    # Data config
    data: str = "ETTh1"
    root_path: str = "data/datasets/ETT-small/"
    data_path: str = "ETTh1.csv"
    features: str = "S"
    target: str = "OT"
    freq: str = "h"
    embed: str = "timeF"

    # Model config
    seq_len: int = 512
    label_len: int = 128
    pred_len: int = 64  # Chronos Bolt max prediction length is 64

    # Training config
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-4
    num_train_epochs: int = 1
    max_steps: int = 10
    eval_steps: int = 5
    save_steps: int = 5
    warmup_steps: int = 0
    logging_steps: int = 1

    # GemmaTS config
    gemma_model_name: str = "google/gemma-3-270m"
    chronos_pretrained: str = "amazon/chronos-bolt-tiny"
    input_patch_size: int = 16
    input_patch_stride: int = 8
    text_prompt: str = "Predict the next values of this time series:"

    # Paths
    output_dir: str = "data/checkpoints/gemma_ts_test"
    seed: int = 42
