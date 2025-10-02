"""Training script for GemmaTS using HuggingFace Trainer."""

import os
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from src.models.gemma_ts import create_gemma_ts
from src.models.chronos_bolt import create_chronos_bolt
from src.dataloader.data_factory import data_provider
from src.utils.metrics import mse, mae, smape
from src.utils.seed import set_seed
from src.configs.test import Config as TestConfig
from src.configs.full_train import Config as FullTrainConfig
from src.configs.chronos import Config as ChronosConfig
from src.configs.chronos_test import Config as ChronosTestConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIGS = {
    "test": TestConfig,
    "full_train": FullTrainConfig,
    "chronos": ChronosConfig,
    "chronos_test": ChronosTestConfig,
}


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Wrapper to make dataloader compatible with HF Trainer."""

    def __init__(self, dataset, pred_len, scaler):
        self.dataset = dataset
        self.pred_len = pred_len
        self.scaler = scaler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.dataset[idx]

        context = torch.from_numpy(seq_x)
        # seq_y contains [label_len + pred_len] values
        # We only want the last pred_len values (pure future, no overlap with context)
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

        outputs = model(context=context, mask=None, target=None, target_mask=None)

        # Get median quantile prediction (0.5)
        quantiles = model.chronos_config.quantiles
        median_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
        preds = outputs.quantile_preds[:, median_idx, :]

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(preds, target)

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

        quantiles = model.chronos_config.quantiles
        median_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()

        # Get scaler from eval dataset
        scaler = eval_dataset.dataset.scaler if eval_dataset else None  # type: ignore[union-attr]

        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                outputs = model(  # type: ignore[misc]
                    context=batch["context"],
                    mask=None,
                    target=None,
                    target_mask=None,
                )

            preds = outputs.quantile_preds[:, median_idx, :].cpu()  # type: ignore[attr-defined]
            target_cpu = batch["target"].cpu()

            # Compute metrics on normalized scale (same as PatchTST)
            # This matches the reference implementation exactly
            loss = torch.nn.functional.mse_loss(preds, target_cpu)
            all_losses.append(loss.item())
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


def main(config_name):
    """Main training loop."""
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}"
        )

    config = CONFIGS[config_name]()
    set_seed(config.seed)

    # Load data
    logger.info("Loading data...")
    train_dataset, _ = data_provider(config, flag="train")
    val_dataset, _ = data_provider(config, flag="val")
    test_dataset, _ = data_provider(config, flag="test")

    # Wrap datasets - pass scaler for inverse transform
    train_ds = TimeSeriesDataset(train_dataset, config.pred_len, train_dataset.scaler)
    val_ds = TimeSeriesDataset(val_dataset, config.pred_len, val_dataset.scaler)
    test_ds = TimeSeriesDataset(test_dataset, config.pred_len, test_dataset.scaler)

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")

    # Initialize model
    if config_name in ["chronos", "chronos_test"]:
        logger.info("Initializing Chronos Bolt baseline model...")
        model = create_chronos_bolt(
            chronos_base=config.chronos_pretrained,
            context_length=config.seq_len,
            prediction_length=config.pred_len,
            patch_size=config.input_patch_size,
            patch_stride=config.input_patch_stride,
        )
    else:
        logger.info("Initializing GemmaTS model...")
        model = create_gemma_ts(
            chronos_base=config.chronos_pretrained,
            gemma_model=config.gemma_model_name,
            context_length=config.seq_len,
            prediction_length=config.pred_len,
            patch_size=config.input_patch_size,
            patch_stride=config.input_patch_stride,
            text_prompt=config.text_prompt,
        )

    # Training arguments
    training_args = TrainingArguments(  # type: ignore
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
        "--config",
        type=str,
        default="full_train",
        help="Config name from configs directory (default: full_train)",
    )
    args = parser.parse_args()

    main(config_name=args.config)
