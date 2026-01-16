# Quick Start Guide

Get your FunctionGemma model fine-tuned in few steps.

## Prerequisites

- TPU v5litepod-8 VM (or compatible TPU)
- Hugging Face account with access to google/functiongemma-270m-it
- ~15 minutes for training

## Step 1: Setup

SSH into your TPU VM. If you don't have uv installed, install it first for faster package installation:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Refer to uv [installation page](https://docs.astral.sh/uv/getting-started/installation/) for details.
You can now install this sub-project.

```bash
uv venv
source .venv/bin/activate
uv pip install finetune
```

This will install all dependencies and provide the `fitnesscoach_finetune` launcher:

The script will:
- Detect and use `uv` if available (much faster!) or fall back to `pip`
- Create a virtual environment at `.venv`
- Install PyTorch 2.9.0 (CPU-only)
- Install torch_xla 2.9.0 with TPU/Pallas support
- Install Hugging Face libraries
- Install the launcher

It is highly recommended to login to Hugging Face to avoid rate limitations:

```bash
huggingface-cli login
```

## Step 2: Train (10-15 minutes)

Make sure your virtual environment is activated:

```bash
source .venv/bin/activate  # If not already activated
```

Start training with defaults:

```bash
fitnesscoach_finetune
```

Or customize:

```bash
fitnesscoach_finetune \
  --num_epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-4
```

**Monitor progress:**

Progress will be reported using [trackio](https://github.com/gradio-app/trackio). It will be accessible through a space created under your HuggingFace profile, or locally by launching:

```bash
# In another terminal
trackio show
```

## What You Get

Your fine-tuned model will be saved to:
```
./output/functiongemma-fitness-coach/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── tokenizer.json            # Tokenizer
└── ...
```

## Expected Results

- **Baseline accuracy:** ~58% on fitness function calling
- **After fine-tuning:** ~85% accuracy
- **Training time:** 10-15 minutes on TPU v5litepod-8
- **Model size:** ~270M parameters + ~8M LoRA parameters

## Common Issues

### "TPU not found"
```bash
python3 -c "import torch_xla; print(torch_xla.device())"
```
Should output `xla:0`. If not, you're not on a TPU VM.

### "Dataset not found"
Make sure you're authenticated:
```bash
huggingface-cli login
```


## Next Steps

- Read `README.md` for detailed configuration options and troubleshooting
- Push to Hugging Face Hub: `model.push_to_hub("your-username/functiongemma-fitness")`

## Cost Estimate

**TPU v5litepod-8 pricing (as of 2026):**
- ~$2.40/hour on-demand
- Training time: 0.25 hours
- **Total cost:** ~$0.60 per training run

Using preemptible TPUs can reduce costs by 70%.
