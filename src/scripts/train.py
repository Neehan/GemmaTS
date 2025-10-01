"""Training script for GemmaTS using HuggingFace Trainer."""

import os
import argparse
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import torch
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from src.models.gemma_ts import create_gemma_ts
from src.dataloader.data_factory import data_provider
from src.utils.metrics import mse, mae, smape
from src.utils.seed import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for GemmaTS training."""

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
    label_len: int = 256
    pred_len: int = 64

    # Training config
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 2e-4
    num_train_epochs: int = 5
    max_steps: int = -1  # -1 means use num_train_epochs instead
    eval_steps: int = 500
    save_steps: int = 500
    warmup_steps: int = 100
    logging_steps: int = 50

    # GemmaTS config
    gemma_model_name: str = "google/gemma-3-270m"
    chronos_pretrained: str = "amazon/chronos-bolt-tiny"
    input_patch_size: int = 16
    input_patch_stride: int = 8

    # Paths
    output_dir: str = "data/checkpoints/gemma_ts_etth1"
    seed: int = 42


config = Config()


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Wrapper to make dataloader compatible with HF Trainer."""

    def __init__(self, dataset, pred_len):
        self.dataset = dataset
        self.pred_len = pred_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.dataset[idx]

        context = torch.from_numpy(seq_x)
        target = torch.from_numpy(seq_y[-self.pred_len :, :])

        if context.shape[-1] == 1:
            context = context.squeeze(-1)
            target = target.squeeze(-1)

        return {"context": context, "target": target}


class GemmaTSTrainer(Trainer):
    """Custom Trainer for GemmaTS."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        """Compute loss for training."""
        context = inputs["context"]
        target = inputs["target"]

        outputs = model(context=context, mask=None, target=target, target_mask=None)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):  # type: ignore[override]
        """Custom evaluation with metrics."""
        eval_dataloader = self.get_eval_dataloader(eval_dataset)  # type: ignore[arg-type]

        model = self.model
        model.eval()  # type: ignore[union-attr]

        all_losses = []
        all_mse = []
        all_mae = []
        all_smape = []

        chronos_cfg = model.module.chronos_config if hasattr(model, "module") else model.chronos_config  # type: ignore[attr-defined,union-attr]
        quantiles = chronos_cfg.quantiles  # type: ignore[attr-defined]
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2  # type: ignore[arg-type]

        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                outputs = model(  # type: ignore[misc]
                    context=batch["context"],
                    mask=None,
                    target=batch["target"],
                    target_mask=None,
                )

            preds = outputs.quantile_preds[:, median_idx, :].cpu()  # type: ignore[attr-defined]
            target_cpu = batch["target"].cpu()

            all_losses.append(outputs.loss.item())  # type: ignore[attr-defined]
            all_mse.append(mse(target_cpu, preds))
            all_mae.append(mae(target_cpu, preds))
            all_smape.append(smape(target_cpu, preds))

        metrics = {
            f"{metric_key_prefix}_loss": sum(all_losses) / len(all_losses),
            f"{metric_key_prefix}_mse": sum(all_mse) / len(all_mse),
            f"{metric_key_prefix}_mae": sum(all_mae) / len(all_mae),
            f"{metric_key_prefix}_smape": sum(all_smape) / len(all_smape),
        }

        self.log(metrics)
        return metrics


def main(test_mode):
    """Main training loop."""
    set_seed(config.seed)

    if test_mode:
        logger.warning("Running in TEST MODE - will only process a few steps")
        config.num_train_epochs = 1
        config.max_steps = 10
        config.eval_steps = 5
        config.save_steps = 5

    # Load data
    logger.info("Loading data...")
    train_dataset, _ = data_provider(config, flag="train")
    val_dataset, _ = data_provider(config, flag="val")
    test_dataset, _ = data_provider(config, flag="test")

    # Wrap datasets
    train_ds = TimeSeriesDataset(train_dataset, config.pred_len)
    val_ds = TimeSeriesDataset(val_dataset, config.pred_len)
    test_ds = TimeSeriesDataset(test_dataset, config.pred_len)

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")

    # Initialize model
    logger.info("Initializing GemmaTS model...")
    model = create_gemma_ts(
        chronos_base=config.chronos_pretrained,
        gemma_model=config.gemma_model_name,
        context_length=config.seq_len,
        prediction_length=config.pred_len,
        patch_size=config.input_patch_size,
        patch_stride=config.input_patch_stride,
    )

    # Training arguments
    training_args = TrainingArguments(  # type: ignore[call-arg]
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
    )

    # Initialize trainer
    trainer = GemmaTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Final evaluation
    logger.info("Final evaluation on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")

    # Save final model
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    logger.info(f"Saved final model to {config.output_dir}/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GemmaTS model with HF Trainer")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (single sample only to verify everything works)",
    )
    args = parser.parse_args()

    main(test_mode=args.test)
