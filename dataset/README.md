# Fitness Coach Function Calling Dataset

This directory contains a script to generate a training dataset for fine-tuning FunctionGemma (or other function-calling models) on Garmin fitness APIs.

## Overview

The `generate_dataset.py` script creates synthetic training examples that map natural language queries to function calls for fitness tracking. The dataset is designed to teach a language model how to invoke the appropriate Garmin fitness API functions based on user queries.

## Supported Functions

The dataset includes examples for the following fitness tracking functions:

| Function | Description |
|----------|-------------|
| `get_steps()` | Get total steps taken today |
| `get_daily_step_goal()` | Get the daily step goal |
| `get_goal_progress()` | Get progress towards daily step goal (%) |
| `get_sleeping_minutes()` | Get total sleep duration in minutes |
| `get_active_minutes()` | Get total active minutes |
| `get_heart_rate()` | Get min and max heart rate |
| `get_body_battery_level()` | Get current body battery level (0-100) |

## Installation

Install the required dependencies:

```bash
uv pip install datasets
```

Or using pip:

```bash
pip install datasets
```

## Usage

### Generate the Dataset

Run the script from the project root:

```bash
python dataset/generate_dataset.py
```

This will generate:
- `dataset/fitness_coach_function_calling/` - HuggingFace Dataset format
- `dataset/fitness_coach_function_calling.json` - JSON format for inspection
- `dataset/fitness_coach_function_calling.jsonl` - JSONL format for training pipelines

### Load the Dataset

Using HuggingFace datasets library:

```python
from datasets import load_from_disk

dataset = load_from_disk("dataset/fitness_coach_function_calling")
print(f"Number of examples: {len(dataset)}")
print(dataset[0])
```

## Dataset Structure

Each example in the dataset contains:

- **system**: System prompt describing available functions and their schemas
- **user**: Natural language query (e.g., "How many steps have I taken today?")
- **assistant**: Function call in JSON format (e.g., `{"name": "get_steps", "arguments": {}}`)
- **function_name**: The name of the function being called
- **function_arguments**: The arguments as a JSON string
- **tools**: Available tools/functions schema in JSON format

### Example

```json
{
  "system": "You are a helpful fitness assistant...",
  "user": "How many steps have I taken today?",
  "assistant": "{\"name\": \"get_steps\", \"arguments\": {}}",
  "function_name": "get_steps",
  "function_arguments": "{}",
  "tools": "[{\"name\": \"get_steps\", ...}]"
}
```

## Dataset Statistics

The generated dataset contains:
- **Total examples**: 42
- **Examples per function**:
  - `get_steps`: 6 examples
  - `get_daily_step_goal`: 4 examples
  - `get_goal_progress`: 7 examples
  - `get_sleeping_minutes`: 6 examples
  - `get_active_minutes`: 6 examples
  - `get_heart_rate`: 6 examples
  - `get_body_battery_level`: 7 examples

## Publishing to HuggingFace Hub

To share your dataset on HuggingFace:

### 1. Install HuggingFace Hub

```bash
uv pip install huggingface_hub
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace access token when prompted.

### 3. Upload the Dataset

```python
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("dataset/fitness_coach_function_calling")

# Push to HuggingFace Hub
dataset.push_to_hub("tengomucho/fitness-coach-function-calling")
```

Or use the command line:

```bash
huggingface-cli upload tengomucho/fitness-coach-function-calling \
  dataset/fitness_coach_function_calling
```

## Fine-tuning FunctionGemma

Once you have the dataset, you can use it to fine-tune FunctionGemma:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# Load your dataset
dataset = load_from_disk("dataset/fitness_coach_function_calling")

# Load FunctionGemma model
model_name = "google/functiongemma-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare your training arguments and trainer
# ... (follow FunctionGemma fine-tuning documentation)
```

## Customizing the Dataset

To add more examples or modify existing ones:

1. Open `generate_dataset.py`
2. Edit the `generate_training_examples()` function
3. Add new query variations to the examples list
4. Run the script again to regenerate the dataset

Example:

```python
examples.extend([
    {
        "user_query": "Your new query here",
        "function_call": {"name": "get_steps", "arguments": {}}
    }
])
```

## Output Formats

### HuggingFace Dataset Format
- Location: `dataset/fitness_coach_function_calling/`
- Best for: Training with HuggingFace Transformers
- Features: Optimized storage, fast loading, supports streaming

### JSON Format
- Location: `dataset/fitness_coach_function_calling.json`
- Best for: Manual inspection, data exploration
- Features: Human-readable, easy to edit

### JSONL Format
- Location: `dataset/fitness_coach_function_calling.jsonl`
- Best for: Training pipelines, data processing tools
- Features: One example per line, easy to stream

## License

This dataset generation script is part of the Fitness Coach project. Use it freely to create training datasets for your fitness assistant models.

## Contributing

To improve the dataset:
1. Add more natural language variations for existing functions
2. Include edge cases and complex queries
3. Add multi-turn conversation examples
4. Test with your fine-tuned model and iterate

## Related Files

- `../fitnesscoach/provider.py` - Contains the actual fitness API function implementations
- `../fitnesscoach/cli.py` - CLI interface for the fitness coach

## Questions?

For issues or questions about the dataset generation, please refer to the main project README or open an issue.
