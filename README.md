# GemmaTS

**Chronos Bolt with Gemma decoder** for time-series forecasting.

## Architecture

```
Input → Chronos Encoder → Gemma → Chronos Output Head → Quantile Predictions
        (scaling/patching)  (decoder)  (forecasting)
```

- **Encoder**: Chronos Bolt's encoder (instance norm → patching → embedding)
- **Decoder**: Gemma (replaces T5)
- **Output**: Chronos Bolt's quantile prediction head

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Hugging Face authentication for gated models (Gemma):
```bash
# Copy the example env file
cp .env.example .env

# Add your HF token to .env
# Get your token from: https://huggingface.co/settings/tokens
# HF_TOKEN=your_token_here
```

3. Request access to Gemma models:
   - Visit https://huggingface.co/google/gemma-3-270m
   - Click "Request access" and wait for approval

## Training

### Single GPU / CPU

Run as a Python module from the project root:

```bash
python -m src.scripts.train
```

### Multi-GPU (HPC cluster)

Uses `accelerate` for distributed training. First configure accelerate:

```bash
accelerate config
```

Then launch training:

```bash
accelerate launch src/scripts/train.py
```

For 4 GPUs with mixed precision (fp16):

```bash
accelerate launch --num_processes=4 --mixed_precision=fp16 src/scripts/train.py
```

Configuration in `src/scripts/train.py`:
```python
config = {
    "gemma_model_name": "google/gemma-3-270m",
    "chronos_pretrained": "amazon/chronos-bolt-tiny",
    "context_length": 512,
    "prediction_length": 64,
    "input_patch_size": 16,
    "input_patch_stride": 8,
    "lr": 2e-4,
    "batch_size": 4,
    "max_steps": 1000,
}
```

## Usage

```python
from src.models.gemma_ts import create_gemma_ts

# Create model
model = create_gemma_ts(
    chronos_base="amazon/chronos-bolt-tiny",
    gemma_model="google/gemma-3-270m",
    context_length=512,
    prediction_length=64,
)

# Forward pass
outputs = model(context, mask, target, target_mask)
loss = outputs["loss"]
predictions = outputs["quantile_preds"]  # (B, num_quantiles, pred_len)

# Autoregressive generation
generated = model.generate(context, steps=5, mask=mask)
```

## Project Structure

```
GemmaTS/
├── src/
│   ├── models/
│   │   └── gemma_ts.py          # Main model (inherits ChronosBolt)
│   ├── data/
│   │   └── datamodule.py        # Dataset loading
│   ├── utils/
│   │   ├── metrics.py           # MSE, MAE, sMAPE
│   │   └── seed.py              # Reproducibility
│   └── scripts/
│       └── train.py             # Training script
├── data/
│   ├── results/                 # Training logs (TSV)
│   └── checkpoints/             # Saved models
├── requirements.txt
└── README.md
```

## Key Features

- **Minimal code**: Inherits from Chronos Bolt, only overrides decoder
- **No duplication**: Reuses Chronos's scaling, patching, and output logic
- **Quantile forecasting**: Outputs 9 quantiles [0.1, 0.2, ..., 0.9]
- **Autoregressive generation**: Uses median quantile for multi-step forecasting

## Outputs

- **Training log**: `data/results/train_log.tsv` (step, split, loss, mse, mae, smape)
- **Checkpoints**: `data/checkpoints/*.pt`