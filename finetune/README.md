# FunctionGemma Fine-tuning for Fitness Coach

Fine-tune Google's FunctionGemma model for fitness-specific function calling using TPU acceleration and Hugging Face tools.

## Overview

This workspace member provides scripts and configuration to fine-tune the [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) model on the [tengomucho/fitness-coach-function-calling](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling) dataset. The result is a specialized model that can accurately map natural language queries about fitness data to the appropriate function calls.

A version of the resulting fine-tuning has been pushed to the hub as adapter [tengomucho/functiongemma-fitness-coach](https://huggingface.co/tengomucho/functiongemma-fitness-coach)

**Quick Start:** See [QUICKSTART.md](./QUICKSTART.md) to get training in under 15 minutes.

### What is FunctionGemma?

FunctionGemma is Google's small language model (270M parameters) designed specifically for function calling tasks. It's part of the Gemma model family and is optimized for:
- Converting natural language to function calls
- Handling tool/API definitions
- JSON schema understanding
- Low-latency inference

### The Training Dataset

The [tengomucho/fitness-coach-function-calling](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling) dataset contains synthetic examples mapping fitness-related queries to function calls:

**Supported Functions:**
- `get_steps()` - Get today's step count
- `get_daily_step_goal()` - Get daily step goal
- `get_step_goal_progress()` - Get progress toward step goal
- `get_sleeping_minutes()` - Get sleep duration
- `get_active_minutes()` - Get active minutes
- `get_heart_rate()` - Get current heart rate
- `get_body_battery_level()` - Get body battery level

**Example:**
```json
{
  "system": "You are a fitness coach assistant...",
  "user": "How many steps have I taken today?",
  "assistant": "{\"name\": \"get_steps\", \"arguments\": {}}",
  "tools": "[{\"name\": \"get_steps\", \"description\": \"...\", ...}]"
}
```

For information about the dataset generation, refer to the `dataset` directory.

## Architecture

**LoRA (Low-Rank Adaptation):**
- Parameter-efficient fine-tuning approach
- Only trains ~8M additional parameters (vs 270M base)
- Rank: 32, Alpha: 64
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Dropout: 0.05

**FSDPv2 (Fully Sharded Data Parallel v2):**
- Distributed training across TPU cores
- Automatic sharding of model and optimizer states
- XLA/SPMD integration for TPU optimization

**Training Configuration:**
- Batch size: 4 per device (gradient accumulation: 2)
- Learning rate: 5e-5
- Epochs: 3 (default)
- Max sequence length: 512 tokens
- Optimizer: [Adafactor](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html)


## Contributing

When making changes:

1. **Testing:** Test on TPU v5litepod-8 before committing
2. **Linting:** Run `ruff check .` and `ruff format .`
3. **Documentation:** Update README.md and QUICKSTART.md
4. **Compatibility:** Maintain Python 3.11 compatibility

## References

- [FunctionGemma Model Card](https://huggingface.co/google/functiongemma-270m-it)
- [Training Dataset](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling)
- [Hugging Face Accelerate TPU Examples](https://github.com/huggingface/accelerate/tree/main/examples)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [FSDP Documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## License

This project inherits the license from the parent fitnesscoach project. FunctionGemma model is subject to Google's Gemma Terms of Use.
