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

### Quick Start

1. **Activate environment** (if using conda):
```bash
conda activate weather
export KMP_DUPLICATE_LIB_OK=TRUE  # Required for macOS
```

2. **Run training** as a Python module from project root:
```bash
# Quick demo (20 epochs)
python -m src.scripts.train --config gemma_ts_demo

# Full training (100 epochs)
python -m src.scripts.train --config gemma_ts
```

### Configuration Files

Configs are in `src/configs/`:
- `gemma_ts_demo.py`: Quick demo (20 epochs)
- `gemma_ts.py`: Full training (100 epochs)
- `chronos_demo.py`: Chronos Bolt baseline demo
- `chronos.py`: Chronos Bolt baseline full training
- `patchtst.py`: PatchTST baseline

### Multi-GPU (HPC cluster)

Uses `accelerate` for distributed training. First configure accelerate:

```bash
accelerate config
```

Then launch training:

```bash
accelerate launch src/scripts/train.py --config gemma_ts
```

For 4 GPUs with mixed precision (fp16):

```bash
accelerate launch --num_processes=4 --mixed_precision=fp16 src/scripts/train.py --config gemma_ts
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
│   │   ├── gemma_ts.py          # Main model (inherits ChronosBolt)
│   │   ├── chronos_bolt.py      # Chronos Bolt baseline
│   │   └── patchtst.py          # PatchTST baseline
│   ├── configs/
│   │   ├── gemma_ts.py          # Full GemmaTS training
│   │   ├── gemma_ts_demo.py     # Quick GemmaTS demo
│   │   ├── chronos.py           # Chronos Bolt full training
│   │   ├── chronos_demo.py      # Chronos Bolt demo
│   │   └── patchtst.py          # PatchTST config
│   ├── dataloader/
│   │   ├── data_loader.py       # Dataset classes
│   │   └── data_factory.py      # Data provider
│   ├── utils/
│   │   ├── metrics.py           # MSE, MAE
│   │   └── seed.py              # Reproducibility
│   └── scripts/
│       └── train.py             # Training script
├── data/
│   ├── datasets/                # Dataset files
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