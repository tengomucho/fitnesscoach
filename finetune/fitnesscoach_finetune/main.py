#!/usr/bin/env python3
"""
Fine-tune FunctionGemma for fitness coach function calling on TPU v5litepod-8.

This script fine-tunes the google/functiongemma-270m-it model using the
tengomucho/fitness-coach-function-calling dataset on TPU with FSDPv2 and LoRA.

Based on: https://github.com/huggingface/accelerate/blob/main/examples/finetune_lm_tpu.py
"""

import json
import os
import shutil

import torch
import torch_xla
import torch_xla.runtime as xr
import typer
from datasets import load_dataset
from peft import LoraConfig
from rich import print
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model
from trl import SFTConfig, SFTTrainer


# Enable SPMD for FSDPv2
xr.use_spmd()

app = typer.Typer(help="Fine-tune FunctionGemma for fitness coach function calling on TPU.")


def format_for_functiongemma(example, tokenizer):
    """
    Format fitness coach dataset for FunctionGemma fine-tuning.

    Based on the official FunctionGemma fine-tuning notebook:
    https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/functiongemma/finetuning-with-functiongemma.ipynb

    Args:
        example: Dataset example with 'user', 'assistant', and 'tools' fields
        tokenizer: The tokenizer with chat template support

    Returns:
        str: Formatted text ready for training
    """
    # Parse the tools from the dataset
    tools_raw = json.loads(example["tools"])

    # Parse the function call from assistant response
    function_call = json.loads(example["assistant"])

    # Wrap tools in the format expected by FunctionGemma template
    # The template expects: [{"type": "function", "function": {...}}]
    tools = [
        {
            "type": "function",
            "function": tool
        }
        for tool in tools_raw
    ]

    # Create the messages in FunctionGemma format following the official notebook
    # Use "developer", "user", and "assistant" roles (not "model")
    messages = [
        {
            "role": "developer",
            "content": example["system"]
        },
        {
            "role": "user",
            "content": example["user"]
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": function_call["arguments"]  # Keep as dict, not JSON string
                    }
                }
            ]
        }
    ]

    # Use apply_chat_template with tools parameter
    try:
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False
        )
        return formatted_text
    except Exception as e:
        print(f"[red]Error formatting example: {e}[/red]")
        print(f"Messages: {messages}")
        print(f"Tools: {tools}")
        print(f"Function call: {function_call}")
        raise

def train(model_id, dataset_id, output_dir, num_epochs, batch_size, learning_rate, max_steps):
    """
    Fine-tune FunctionGemma on TPU with FSDPv2 and LoRA.

    Args:
        model_id: HuggingFace model ID (e.g., google/functiongemma-270m-it)
        dataset_id: HuggingFace dataset ID or local path
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate for training
        max_steps: Maximum number of training steps
    """

    print(f"Loading model: {model_id}")
    # Load model with low CPU memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,  # Let FSDP handle device placement
    )

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a padding token if necessary
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer))

    print(f"Loading dataset: {dataset_id}")

    # Load dataset
    dataset = load_dataset(dataset_id, split="train")

    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")

    # Show a sample formatted example
    print("\n" + "=" * 60)
    print("Sample formatted example:")
    print("=" * 60)
    sample_text = format_for_functiongemma(dataset[0], tokenizer)
    print(sample_text)
    print("=" * 60 + "\n")

    # Split dataset into train and test sets
    print("\nSplitting dataset into train and test sets (70/30 split, shuffled)...")
    dataset = dataset.train_test_split(test_size=0.3, shuffle=True, seed=42)
    print(f"  Train set: {len(dataset['train'])} examples")
    print(f"  Test set: {len(dataset['test'])} examples")

    # Identify transformer layer class for FSDP wrapping
    # FunctionGemma is based on Gemma architecture
    transformer_layer_cls_to_wrap = model.model.layers[0].__class__.__name__
    print(f"Wrapping transformer layer: {transformer_layer_cls_to_wrap}")

    # Configure FSDP training arguments for TPU
    fsdp_training_args = {
        "fsdp": "full_shard",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [transformer_layer_cls_to_wrap],
            "xla": True,
            "xla_fsdp_v2": True,
            "xla_fsdp_grad_ckpt": True,
        },
    }

    # Set up PEFT LoRA configuration
    # Target modules for FunctionGemma (based on Gemma architecture)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # Configure SFT training parameters
    sft_config = SFTConfig(
        # Training hyperparameters
        gradient_checkpointing=False,  # Disabled when using xla_fsdp_grad_ckpt
        max_length=512,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        max_steps=max_steps,

        # Optimizer and scheduler
        optim="adafactor",
        lr_scheduler_type="constant",

        # Logging and saving
        output_dir=output_dir,
        logging_steps=1,
        # save_strategy="epoch",
        eval_strategy="epoch",

        # Dataset configuration
        dataset_text_field="text",
        packing=False,  # Disable packing for function calling

        # Miscellaneous
        dataloader_drop_last=True,
        bf16=True,
        save_safetensors=True,
        report_to="trackio",

        # FSDP configuration
        **fsdp_training_args,
    )

    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Output directory: {output_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA r: {lora_config.r}")
    print(f"LoRA alpha: {lora_config.lora_alpha}")
    print(f"LoRA dropout: {lora_config.lora_dropout}")
    print(f"Target modules: {lora_config.target_modules}")
    print("=" * 60 + "\n")

    # Initialize and run trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=lambda example: format_for_functiongemma(example, tokenizer),
    )

    trainer._save_tpu

    print("Starting training...")
    trainer.train()

    print("\nTraining complete!")

    # Save the final LoRA adapter model and tokenizer
    print("Saving LoRA adapter and tokenizer...")

    # This ensures all adapter weights are gathered and saved correctly after unwrapping the model on the TPU when
    # using FSDPv2. This is a workaround, see here: https://github.com/huggingface/transformers/issues/36004#issuecomment-2635144319
    torch_xla.sync()
    unwrapped_model = unwrap_model(trainer.model, recursive=True).to("cpu")
    unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("🥳 Fine-tuning Complete!")
    print("=" * 60)
    print(f"Model and tokenizer saved to: {output_dir}")
    print("\n🔥 To use your fine-tuned model:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print()
    print("To upload the model to the Hugging Face Hub, run:")
    print("   model.push_to_hub('tenfomucho/functiongemma-fitness')")
    print("=" * 60)


def check_setup(model_id: str):
    """
    Verify the system is correctly setup for training.
    """

    print("Checking system setup...")

    # First check if we are running on a TPU
    num_devices = torch_xla.device_count()
    xm = torch_xla.core.xla_model
    device = torch_xla.device()
    type = xm.xla_device_hw(device)
    if "TPU" != type:
        raise ValueError("❌ Not running on a TPU")
    print(f"✅ Running on a {type} with {num_devices} devices")

    # Now check if we are authenticated with Hugging Face, to access the model
    try:
        AutoConfig.from_pretrained(model_id)
        print(f"✅ Authenticated with 🤗 Hugging Face for model: [bold green]{model_id}[/bold green]")
    except Exception as e:
        print(f"❌ Cannot access 🤗 Hugging Face model: [bold red]{model_id}[/bold red]")
        raise e


@app.command("train")
def train_command(
    model_id: str = typer.Option(
        "google/functiongemma-270m-it",
        "--model-id", "-m",
        help="Model ID to use for training"
    ),
    dataset_id: str = typer.Option(
        "tengomucho/fitness-coach-function-calling",
        "--dataset-id", "-d",
        help="Dataset ID from HuggingFace Hub or local path"
    ),
    output_dir: str = typer.Option(
        "./output/functiongemma-fitness-coach",
        "--output-dir", "-o",
        help="Output directory for fine-tuned model"
    ),
    num_epochs: int = typer.Option(
        8,
        "--num-epochs", "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size", "-b",
        help="Per-device batch size"
    ),
    learning_rate: float = typer.Option(
        5e-5,
        "--learning-rate", "-lr",
        help="Learning rate"
    ),
    max_steps: int = typer.Option(
        -1,
        "--max-steps", "-s",
        help="Maximum number of training steps, -1 for no limit"
    ),
):
    """Fine-tune FunctionGemma for fitness coach function calling on TPU."""

    check_setup(model_id)

    print()
    print("Training configuration:")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Output directory: {output_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists")
        overwrite = typer.confirm("Do you want to overwrite it?", default=False, abort=False)
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            print("Aborting training to avoid overwriting existing model")
            return

    # Ask to type return to continue
    typer.confirm("Are you sure you want to continue?", default=True, abort=True)

    print("Starting training...")

    train(
        model_id=model_id,
        dataset_id=dataset_id,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    app()
