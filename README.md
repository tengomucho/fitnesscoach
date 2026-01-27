# Fitness Coach

This is a project that shows how it is possible to get a simple fitness coach assistant.

## Installation

To build and install the project, make sure you have Python 3.10 or newer. Then run:

```bash
pip install .
```

Or, to install in editable/development mode:

```bash
pip install -e .
```

All dependencies will be installed automatically.

## Usage

After installation, you can use the CLI tool. To get a daily fitness summary, run:

```bash
fitnesscoach summary
```

This command fetches your latest health and fitness data and outputs a concise summary.

To chat with the Fitness Coach assistant, run:

```bash
fitnesscoach chat
```

You can also ask a specific question directly:

```bash
fitnesscoach chat --question "What was my performance today?"
```

This command allows you to interact with the fitness coach AI, which can analyze your fitness data and provide personalized insights. Use the `--verbose` flag to see detailed function calls, and `--model-id` to specify a different model.

You can get help and see all commands with:

```bash
fitnesscoach --help
```

## Login

This app uses data that is fetched from Garmin Connect, using the excellent `garminconnect` package. In order to be able to get the data, you need to login once, by doing

```bash
fitnesscoach login
```

## Fine-tuning FunctionGemma

This project includes scripts to fine-tune Google's FunctionGemma model on the fitness coach function calling dataset using TPU.

**Quick start on TPU v5litepod-8:**

```bash
uv pip install finetune
```

**What you get:**
- Fine-tuned FunctionGemma model for fitness function calling
- LoRA adapters for parameter-efficient training

**Documentation:**
- `finetune/README.md` - Complete guide and configuration options
- Dataset: [tengomucho/fitness-coach-function-calling](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling)

## Run tests

To run tests you can just run

```bash
pytest -sv tests
```
