#!/usr/bin/env python3
"""
Generate a training dataset for fine-tuning FunctionGemma on Garmin fitness APIs.

This script creates synthetic examples of user queries and corresponding function calls
for the following fitness tracking functions:
- get_steps
- get_daily_step_goal
- get_goal_progress
- get_sleeping_minutes
- get_active_minutes
- get_heart_rate
- get_body_battery_level
"""

import json
import os
import shutil
from typing import Any

import typer
from datasets import Dataset


# Define function schemas compatible with function calling format
FUNCTION_DEFINITIONS = [
    {
        "name": "get_steps",
        "description": "Get the total number of steps taken today from the Garmin watch.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_daily_step_goal",
        "description": "Get the daily step goal set on the Garmin watch.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_goal_progress",
        "description": "Get the progress towards the daily step goal as a percentage.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_sleeping_minutes",
        "description": "Get the total minutes of sleep recorded today from the Garmin watch.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_active_minutes",
        "description": "Get the total active minutes recorded today from the Garmin watch.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_heart_rate",
        "description": "Get the minimum and maximum heart rate recorded today from the Garmin watch. Returns a tuple of (min_heart_rate, max_heart_rate).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_body_battery_level",
        "description": "Get the most recent body battery level from the Garmin watch. Body battery is a measure of energy reserves, ranging from 0 to 100.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def create_system_prompt() -> str:
    """Create a system prompt that describes available functions."""
    return f"""You are a helpful fitness assistant with access to Garmin fitness data. You can help users track their health and fitness metrics.

You have access to the following functions:

{json.dumps(FUNCTION_DEFINITIONS, indent=2)}

When a user asks about their fitness data, call the appropriate function(s) to retrieve the information."""


def generate_training_examples() -> list[dict[str, Any]]:
    """Generate synthetic training examples for function calling."""

    examples = []

    # Examples for get_steps
    examples.extend(
        [
            {
                "user_query": "How many steps have I taken today?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {"user_query": "What's my step count?", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Show me my steps for today", "function_call": {"name": "get_steps", "arguments": {}}},
            {
                "user_query": "Can you tell me how many steps I've walked?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {"user_query": "Steps today?", "function_call": {"name": "get_steps", "arguments": {}}},
        ]
    )

    # Examples for get_daily_step_goal
    examples.extend(
        [
            {
                "user_query": "What's my daily step goal?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "How many steps am I supposed to take today?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "What's my step target?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {"user_query": "Tell me my step goal", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
        ]
    )

    # Examples for get_goal_progress
    examples.extend(
        [
            {
                "user_query": "How close am I to my step goal?",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
            {
                "user_query": "What's my progress towards my daily goal?",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Am I close to reaching my step target?",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Show me my goal progress",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Have I hit my step goal yet?",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
            {
                "user_query": "What percentage of my step goal have I completed?",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
        ]
    )

    # Examples for get_sleeping_minutes
    examples.extend(
        [
            {
                "user_query": "How much did I sleep last night?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "What's my sleep duration?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "How many hours of sleep did I get?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Show me my sleep time",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Tell me about my sleep",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
        ]
    )

    # Examples for get_active_minutes
    examples.extend(
        [
            {
                "user_query": "How active have I been today?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "What's my activity time?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "How many active minutes do I have?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Show me my active time for today",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Tell me my activity minutes",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
        ]
    )

    # Examples for get_heart_rate
    examples.extend(
        [
            {
                "user_query": "What was my heart rate today?",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "Show me my heart rate range",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {"user_query": "What's my heart rate data?", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {
                "user_query": "Tell me my min and max heart rate",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "What's my heart rate been like today?",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
        ]
    )

    # Examples for get_body_battery_level
    examples.extend(
        [
            {
                "user_query": "What's my body battery?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "How much energy do I have?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Show me my body battery level",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "What's my current energy level?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Tell me my body battery",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "How recharged am I?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
        ]
    )

    # More complex examples with conversational context
    examples.extend(
        [
            {
                "user_query": "I want to know if I've been active enough today",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Did I reach my step target today?",
                "function_call": {"name": "get_goal_progress", "arguments": {}},
            },
            {
                "user_query": "I'm feeling tired, can you check my body battery?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "I want a quick summary of my steps",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {
                "user_query": "Was my sleep good last night?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Check my cardiovascular activity from today",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
        ]
    )

    return examples


def format_for_function_calling(examples: list[dict[str, Any]]) -> dict[str, list]:
    """Format examples for function calling training.

    Creates a dataset with the following columns:
    - system: System prompt describing available functions
    - user: User's query
    - assistant: Assistant's response with function call in JSON format
    - function_name: The name of the function being called
    - function_arguments: The arguments as JSON string
    """

    system_prompt = create_system_prompt()

    formatted_data = {
        "system": [],
        "user": [],
        "assistant": [],
        "function_name": [],
        "function_arguments": [],
        "tools": [],  # Store the available tools for each example
    }

    for example in examples:
        formatted_data["system"].append(system_prompt)
        formatted_data["user"].append(example["user_query"])

        # Format assistant response as function call
        function_call = example["function_call"]
        assistant_response = json.dumps(function_call)
        formatted_data["assistant"].append(assistant_response)

        # Add function details separately for easier filtering/analysis
        formatted_data["function_name"].append(function_call["name"])
        formatted_data["function_arguments"].append(json.dumps(function_call["arguments"]))

        # Add available tools
        formatted_data["tools"].append(json.dumps(FUNCTION_DEFINITIONS))

    return formatted_data


def cleanup_generated_files():
    """Remove all generated dataset files and directories."""
    files_to_remove = [
        "dataset/fitness_coach_function_calling.json",
        "dataset/fitness_coach_function_calling.jsonl",
    ]
    dirs_to_remove = ["dataset/fitness_coach_function_calling"]

    print("Cleaning up generated files...")

    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  Removed: {file_path}")
        else:
            print(f"  Not found: {file_path}")

    # Remove directories
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  Removed directory: {dir_path}")
        else:
            print(f"  Not found: {dir_path}")

    print("\nCleanup complete!")


def main():
    """Generate and save the dataset."""
    print("Generating training examples...")
    examples = generate_training_examples()
    print(f"Generated {len(examples)} training examples")

    print("\nFormatting examples for function calling...")
    formatted_data = format_for_function_calling(examples)

    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_dict(formatted_data)

    print("\nDataset info:")
    print(f"  Number of examples: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")

    # Print a few examples
    print("\nSample examples:")
    for i in range(min(3, len(dataset))):
        print(f"\n--- Example {i + 1} ---")
        print(f"User: {dataset[i]['user']}")
        print(f"Assistant: {dataset[i]['assistant']}")
        print(f"Function: {dataset[i]['function_name']}")

    # Save to disk in multiple formats
    output_dir = "dataset/fitness_coach_function_calling"

    print(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    print("Saved dataset to disk format")

    # Also save as JSON for easy inspection
    json_path = "dataset/fitness_coach_function_calling.json"
    print(f"Saving dataset to {json_path}...")
    with open(json_path, "w") as f:
        json.dump(formatted_data, f, indent=2)
    print("Saved dataset to JSON format")

    # Save as JSONL (common format for training)
    jsonl_path = "dataset/fitness_coach_function_calling.jsonl"
    print(f"Saving dataset to {jsonl_path}...")
    with open(jsonl_path, "w") as f:
        for i in range(len(dataset)):
            row = {
                "system": dataset[i]["system"],
                "user": dataset[i]["user"],
                "assistant": dataset[i]["assistant"],
                "function_name": dataset[i]["function_name"],
                "function_arguments": dataset[i]["function_arguments"],
                "tools": dataset[i]["tools"],
            }
            f.write(json.dumps(row) + "\n")
    print("Saved dataset to JSONL format")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total examples: {len(dataset)}")

    # Count examples per function
    from collections import Counter

    function_counts = Counter(dataset["function_name"])
    print("\nExamples per function:")
    for func_name, count in sorted(function_counts.items()):
        print(f"  {func_name}: {count}")

    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print("=" * 50)
    print("\nFiles created:")
    print(f"  1. {output_dir}/ (HuggingFace Dataset format)")
    print(f"  2. {json_path} (JSON format)")
    print(f"  3. {jsonl_path} (JSONL format)")
    print("\nYou can now upload these to HuggingFace Hub!")
    print("\nTo upload to HuggingFace:")
    print("  1. Install huggingface_hub: pip install huggingface_hub")
    print("  2. Login: huggingface-cli login")
    print("  3. Upload: dataset.push_to_hub('your-username/fitness-coach-function-calling')")


app = typer.Typer(help="Generate training dataset for FunctionGemma or cleanup generated files.")


@app.command()
def generate(
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Remove all generated dataset files and directories",
    ),
):
    """Generate training dataset for FunctionGemma or cleanup generated files."""
    if cleanup:
        cleanup_generated_files()
    else:
        main()


if __name__ == "__main__":
    app()
