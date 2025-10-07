"""Training script for GemmaTS using HuggingFace Trainer."""

import os
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Save results
import json
import numpy as np
import torch
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from src.models.gemma_ts import create_gemma_ts
from src.models.chronos_bolt import create_chronos_bolt
from src.models.patchtst import create_patchtst
from src.dataloader.data_factory import data_provider
from src.utils.metrics import mse, mae
from src.utils.losses import mse_loss, quantile_loss
from src.utils.seed import set_seed
from src.configs.gemma_ts import Config as GemmaTSConfig
from src.configs.gemma_ts_demo import Config as GemmaTSDemoConfig
from src.configs.chronos import Config as ChronosConfig
from src.configs.chronos_demo import Config as ChronosDemoConfig
from src.configs.patchtst import Config as PatchTSTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIGS = {
    "gemma_ts": GemmaTSConfig,
    "gemma_ts_demo": GemmaTSDemoConfig,
    "chronos": ChronosConfig,
    "chronos_demo": ChronosDemoConfig,
    "patchtst": PatchTSTConfig,
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
    """Custom Trainer for time series models."""

    def __init__(self, model_type, loss_fn, *args, **kwargs):
        """
        Args:
            model_type: One of 'chronos', 'patchtst', or 'gemma'
            loss_fn: Loss function to use ('mse' or 'quantile')
        """
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.loss_fn = loss_fn

    def _get_unwrapped_model(self, model):
        """Get the underlying model from DataParallel/DistributedDataParallel wrapper."""
        return model.module if hasattr(model, "module") else model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        """Compute loss for training."""
        context = inputs["context"]
        target = inputs["target"]

        outputs = model(context=context, mask=None, target=None, target_mask=None)

        # Compute loss based on configured loss function
        if self.loss_fn == "mse":
            if self.model_type == "patchtst":
                preds = outputs.predictions
            else:
                unwrapped_model = self._get_unwrapped_model(model)
                quantiles = unwrapped_model.chronos_config.quantiles
                median_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
                preds = outputs.quantile_preds[:, median_idx, :]
            loss = mse_loss(preds, target)
        elif self.loss_fn == "quantile":
            unwrapped_model = self._get_unwrapped_model(model)
            quantiles = unwrapped_model.chronos_config.quantiles
            loss = quantile_loss(outputs.quantile_preds, target, quantiles)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):  # type: ignore[override]
        """Custom evaluation with metrics."""
        eval_dataloader = self.get_eval_dataloader(eval_dataset)  # type: ignore[arg-type]

        model = self.model
        model.eval()  # type: ignore[union-attr]

        all_preds = []
        all_targets = []

        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                outputs = model(  # type: ignore[misc]
                    context=batch["context"],
                    mask=None,
                    target=None,
                    target_mask=None,
                )

            # Get predictions based on model type
            if self.model_type == "patchtst":
                preds = outputs.predictions
            else:  # chronos
                unwrapped_model = self._get_unwrapped_model(model)
                quantiles = unwrapped_model.chronos_config.quantiles  # type: ignore[union-attr]
                median_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
                preds = outputs.quantile_preds[:, median_idx, :]

            all_preds.append(preds)
            all_targets.append(batch["target"])

        # Move to CPU once at the end
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_targets = torch.cat(all_targets, dim=0).cpu()

        # Compute metrics on normalized scale
        loss = torch.nn.functional.mse_loss(all_preds, all_targets)
        mse_val = mse(all_targets, all_preds)
        mae_val = mae(all_targets, all_preds)

        metrics = {
            f"{metric_key_prefix}_loss": loss.item(),
            f"{metric_key_prefix}_mse": mse_val,
            f"{metric_key_prefix}_mae": mae_val,
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
    train_ds = TimeSeriesDataset(train_dataset, config.pred_len, train_dataset.scaler)  # type: ignore[attr-defined]
    val_ds = TimeSeriesDataset(val_dataset, config.pred_len, val_dataset.scaler)  # type: ignore[attr-defined]
    test_ds = TimeSeriesDataset(test_dataset, config.pred_len, test_dataset.scaler)  # type: ignore[attr-defined]

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")

    # Initialize model
    if config_name in ["chronos", "chronos_demo"]:
        logger.info("Initializing Chronos Bolt baseline model...")
        model = create_chronos_bolt(
            chronos_base=config.chronos_pretrained,
            context_length=config.seq_len,
            prediction_length=config.pred_len,
            patch_size=config.input_patch_size,
            patch_stride=config.input_patch_stride,
            freeze=config.freeze,
            use_bfloat16=config.use_bfloat16,
        )
    elif config_name == "patchtst":
        logger.info("Initializing PatchTST baseline model...")
        model = create_patchtst(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            enc_in=config.enc_in,
            e_layers=config.e_layers,
            n_heads=config.n_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            fc_dropout=config.fc_dropout,
            head_dropout=config.head_dropout,
            patch_len=config.patch_len,
            stride=config.stride,
        )
    else:
        logger.info(
            f"Initializing GemmaTS model ({config.chronos_pretrained} + {config.gemma_model_name})..."
        )
        model = create_gemma_ts(
            chronos_base=config.chronos_pretrained,
            gemma_model=config.gemma_model_name,
            context_length=config.seq_len,
            prediction_length=config.pred_len,
            patch_size=config.input_patch_size,
            patch_stride=config.input_patch_stride,
            text_prompt=config.text_prompt,
            freeze=config.freeze,
            use_bfloat16=config.use_bfloat16,
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
        eval_strategy="steps",
        save_strategy="no",  # Don't save checkpoints during training
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
    )

    # Determine model type
    if config_name in ["chronos", "chronos_demo"]:
        model_type = "chronos"
    elif config_name == "patchtst":
        model_type = "patchtst"
    else:
        model_type = "gemma"

    # Initialize trainer
    trainer = GemmaTSTrainer(
        model_type=model_type,
        loss_fn=config.loss_fn,
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

    results_dir = os.path.join("data/results", config_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics as JSON
    metrics_file = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test metrics to {metrics_file}")

    # Save metrics as text for easy viewing
    results_txt = os.path.join(results_dir, "results.txt")
    with open(results_txt, "w") as f:
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write(f"Dataset: {config.data} (features={config.features})\n")
        f.write(f"seq_len={config.seq_len}, pred_len={config.pred_len}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Test Set Results (Normalized Scale)\n")
        f.write("=" * 50 + "\n")
        for key, value in test_metrics.items():
            if key != "epoch":
                f.write(f"{key}: {value:.6f}\n")
    logger.info(f"Saved results summary to {results_txt}")

    # Save predictions (optional, can be large)
    save_predictions = os.getenv("SAVE_PREDICTIONS", "false").lower() == "true"
    if save_predictions:
        logger.info("Saving predictions...")
        all_preds = []
        all_targets = []

        model.eval()  # type: ignore[union-attr]
        test_dataloader = trainer.get_test_dataloader(test_ds)

        for batch in test_dataloader:
            batch = trainer._prepare_inputs(batch)
            with torch.no_grad():
                outputs = model(  # type: ignore[misc]
                    context=batch["context"],
                    mask=None,
                    target=None,
                    target_mask=None,
                )

            # Get predictions based on model type
            if model_type == "patchtst":
                preds = outputs.predictions
            else:  # chronos or gemma
                unwrapped_model = trainer._get_unwrapped_model(model)  # type: ignore[attr-defined]
                quantiles = unwrapped_model.chronos_config.quantiles  # type: ignore[union-attr]
                median_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
                preds = outputs.quantile_preds[:, median_idx, :]

            all_preds.append(preds)
            all_targets.append(batch["target"])

        # Move to CPU once at the end
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()

        np.save(os.path.join(results_dir, "predictions.npy"), all_preds)
        np.save(os.path.join(results_dir, "targets.npy"), all_targets)
        logger.info(f"Saved predictions to {results_dir}")

    # Optionally save final model (disabled by default due to size)
    save_final = os.getenv("SAVE_FINAL_MODEL", "false").lower() == "true"
    if save_final:
        trainer.save_model(os.path.join(config.output_dir, "final_model"))
        logger.info(f"Saved final model to {config.output_dir}/final_model")
    else:
        logger.info("Skipping final model save (set SAVE_FINAL_MODEL=true to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GemmaTS model with HF Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="gemma_ts",
        help="Config name from configs directory (default: gemma_ts)",
    )
    args = parser.parse_args()

    main(config_name=args.config)
