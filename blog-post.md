# Fine-Tuning FunctionGemma on TPU to Create a Virtual Fitness Coach in 10 Minutes, $0.50

While most FunctionGemma fine-tuning tutorials focus on GPU setups, Google Cloud TPUs offer a compelling alternative: **faster training, lower cost, and easier scaling**. But TPUs come with unique constraints—dynamic tensor shapes kill performance, and standard training configurations won't work out of the box.

This post demonstrates fine-tuning [FunctionGemma](https://huggingface.co/google/functiongemma-270m-it) (270M parameters) on **TPU v5litepod-8** for a function-calling fitness coach application, achieving **~10 minute training time for ~$0.50**. I'll share the TPU-specific optimizations that made this possible and show how the fine-tuned model significantly reduces hallucinations compared to the base model.

## What You'll Learn

**Key Technical Contributions**:
- **TPU optimization strategies**: Why `pad_to_multiple_of=max_length` reduced training time from ~1 hour to ~10 minutes (6x speedup)
- **FSDP v2 configuration**: Using `xla_fsdp_v2` and `xla_fsdp_grad_ckpt` for efficient multi-device training on TPU
- **Synthetic dataset generation**: Creating 213 function-calling examples formatted for FunctionGemma
- **Quantitative evaluation**: Comparing base model vs. fine-tuned performance on hallucination rates

**Why TPU for Small Model Fine-Tuning?**:
- **Cost-effective**: ~$0.50 total training cost (vs. $5-10 on GPU cloud instances)
- **Fast iteration**: 10-minute training enables rapid experimentation

**Project Resources**:
- Complete code: [github.com/tengomucho/fitnesscoach](https://github.com/tengomucho/fitnesscoach)
- Fine-tuned model: [tengomucho/functiongemma-fitness-coach](https://huggingface.co/tengomucho/functiongemma-fitness-coach)
- Training dataset: [tengomucho/fitness-coach-function-calling](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling)

## Project Overview

The goal was to create a virtual fitness coach that can answer questions about fitness data retrieved from a fitness device, in my case my Garmin watch. The project follows these steps:

1. **Define the API**: Python functions to retrieve fitness data (steps, sleep, heart rate, etc.)
2. **Generate synthetic dataset**: 213 training examples mapping user queries to function calls
3. **Fine-tune on TPU**: Optimize FunctionGemma with TPU-specific configurations
4. **Build the chat interface**: Interactive CLI that runs inference and executes tool calls

Let's dive into the details.

## Defining the Provider API and Testing

Before teaching a model to call functions, you need functions to call. The first step was defining a simple API layer that wraps Garmin Connect data retrieval—these would become the "tools" available to our fitness coach.

I used the handy `garminconnect` Python module to retrieve data from my Garmin watch and defined functions to access specific metrics. For example, here's how to retrieve today's step count:

```python
def get_steps() -> int:
    """Get the number of steps walked today.

    Args:
        None

    Returns:
        int: The number of steps taken today
    """
    summary = get_summary()
    return summary.get("totalSteps", 0)
```

I followed the same pattern to define seven functions in total:

- `get_steps` - Today's step count
- `get_daily_step_goal` - Target steps for the day
- `get_step_goal_progress` - Progress percentage toward goal
- `get_sleeping_minutes` - Duration of sleep last night
- `get_active_minutes` - Active time today
- `get_heart_rate` - Current heart rate
- `get_body_battery_level` - Garmin's energy level metric (0-100)

To verify the API worked correctly, I built a simple CLI that retrieves and displays the data:

```bash
$ fitnesscoach summary
Steps: 2400 (31.41%)
Daily Step Goal: 7640
Active time: 1h:3m
Sleeping time: 7h:13m
```

With the API working, the next challenge was teaching the model when and how to call these functions.

## Creating a Dataset for Fine Tuning

FunctionGemma expects a specific message format for function calling. The model uses three roles —`developer`, `user`, and `assistant`— in a structured conversation format:

```python
message = [
    {
        # System prompt
        "role": "developer",
        "content": "You are a model that can do function calling with the following functions"
    },
    {
        "role": "user",
        "content": "Tell me the steps count"
    }
]
```

The messages can be processed through the common chat template defined alongside the model:


```python
tools = [get_steps]
processor = AutoProcessor.from_pretrained(MODEL_ID, device_map="auto")
inputs = processor.apply_chat_template(
    message,
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
```

You can then use these inputs with `model.generate` to get a function call, parse it, execute the function, and provide the result back to the model for a final natural language response. Here's what the complete conversation flow looks like:

```python
[
    {'role': 'developer', 'content': 'You are a model that can do function calling with the following functions'},
    {'role': 'user', 'content': 'tell me the steps walked'},
    {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_steps', 'arguments': {}}}]},
    {'role': 'tool', 'name': 'get_steps', 'content': '2400'},
    {'role': 'assistant', 'content': 'The number of steps walked was 2400.'}
]
```

For complete implementation details, check the FunctionGemma [model card](https://huggingface.co/google/functiongemma-270m-it).

### Building the Training Dataset

The goal was creating a synthetic dataset mapping natural language queries to the correct function calls. I wrote a script that generated 213 training examples covering various ways users might ask about their fitness data:

```json
{"user_query": "My walking steps", "function_call": {"name": "get_steps", "arguments": {}}},
{"user_query": "Number of steps taken", "function_call": {"name": "get_steps", "arguments": {}}},
{"user_query": "Show steps", "function_call": {"name": "get_steps", "arguments": {}}},
{"user_query": "Get my steps", "function_call": {"name": "get_steps", "arguments": {}}},
```

The beauty of this approach is that `get_json_schema` from `transformers.utils` automatically extracts function signatures and docstrings from the Python code, generating the JSON tool definitions FunctionGemma needs. No manual schema writing required.

I published the complete dataset as [fitness-coach-function-calling](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling) on Hugging Face Hub, making it easy to reproduce the fine-tuning process.

## Fine Tuning the Model on TPU

With the dataset ready, it was time for the compute-intensive part: fine-tuning the model. While you can run FunctionGemma inference on a typical PC, training requires significant compute and memory. This is where Google's TPUs shine — but they require specific optimizations to achieve peak performance.

Here's what I learned the hard way: without proper configuration, TPU training can actually be *slower* than GPU due to repeated graph compilation. This section covers the TPU setup and three critical optimizations that reduced training time from ~1 hour to ~10 minutes.

### Setting Up Your TPU Environment

Provision a TPU v5litepod-8:
```bash
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version v2-alpha-tpuv5-lite
```

Connect to your TPU via SSH:
```bash
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-west4-a
```

For the simplest setup, clone the project repository and use `uv` to install dependencies from the `finetune` sub-project:

```bash
# Install Astral's uv
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tengomucho/fitnesscoach
cd fitnesscoach
uv venv
source .venv/bin/activate
uv pip install ./finetune
```

**Note**: If you're adapting this for your own project, you'll need: `torch~=2.9.0`, `torch_xla[tpu]~=2.9.0`, `transformers`, `datasets`, `peft`, and `accelerate`.

If you want to reproduce my exact training run, you can launch it with the CLI command `fitnesscoach_finetune`. The following sections explain how the training script works and why each configuration choice matters.

**Common TPU Issues**:

- **Slow training with "Compiling..." messages?** Ensure `pad_to_multiple_of=max_length` in your configuration (see TPU Optimization #3 below).
- **Out of memory error?** Reduce `per_device_train_batch_size` in the training configuration.
- **Slow training over the first steps?** This is normal, as Torch XLA will trace the graphs and this takes a while, but once done the training loop will be much faster.


### TPU Optimization #1: SPMD Initialization

The first optimization is simple but mandatory. Enable Single Program Multiple Data (SPMD) mode before loading the model—this is required for FSDP v2 on TPU:

```python
import torch_xla.runtime as xr
xr.use_spmd()  # Must call before model initialization
```

⚠️ **Important**: Call `use_spmd()` before any model operations. Calling it after model initialization will cause errors.

### Loading the Model

With SPMD enabled, load the model using the standard `transformers` API:

```python
# Load model with low CPU memory usage
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_cache=False,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map=None,  # Let FSDP handle device placement
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset(dataset_id, split="train")
```

### TPU Optimization #2: FSDP v2 Configuration

Configure Fully Sharded Data Parallel v2 (FSDP) for multi-chip TPU training. This is **TPU-specific** and differs from GPU FSDP:

```python
transformer_layer_cls_to_wrap = model.model.layers[0].__class__.__name__

fsdp_training_args = {
    "fsdp": "full_shard",
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": [transformer_layer_cls_to_wrap],
        "xla": True,                    # Enable XLA compilation
        "xla_fsdp_v2": True,           # Use FSDP v2 (required for TPU)
        "xla_fsdp_grad_ckpt": True,    # TPU-specific gradient checkpointing
    },
}
```

**Key differences from GPU**:
- `xla_fsdp_v2=True` is required for TPU (standard FSDP won't work)
- Use `xla_fsdp_grad_ckpt=True` instead of `gradient_checkpointing=True`
- The standard `gradient_checkpointing` parameter must be set to `False` (see below)

### LoRA Configuration

To keep memory usage low and training fast, I used LoRA (Low-Rank Adaptation) to fine-tune only a small subset of parameters:

```python
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

This targets the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) which have the most impact on function calling performance.

### TPU Optimization #3: Static Tensor Shapes (Critical!)

This is the **most important optimization** for TPU performance—and the one that took me the longest to discover:

```python
sft_config = SFTConfig(
    # TPU Critical: Static shapes prevent graph recompilation
    pad_to_multiple_of=max_length,      # ← 6x speedup!
    dataloader_drop_last=True,           # Drop incomplete batches

    # TPU Critical: Gradient checkpointing
    gradient_checkpointing=False,        # Must be False on TPU
                                        # (use xla_fsdp_grad_ckpt instead)

    # Training hyperparameters
    max_length=max_length,
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
    eval_strategy="epoch",
    report_to="trackio",  # Optional: track metrics with trackio

    # Dataset configuration
    dataset_text_field="text",
    packing=False,  # Disable packing for function calling
    bf16=True,

    # FSDP configuration
    **fsdp_training_args,
)
```

**Why `pad_to_multiple_of=max_length` matters**:

TPUs compile a computation graph based on tensor shapes. When a new shape is encountered, TPU must recompile the graph, adding ~30-60 seconds of overhead **per unique shape**. With variable-length sequences, every batch can have a different shape, causing constant recompilation.

**The impact**:
- ❌ **Without padding**: Training took ~60+ minutes (spent most time recompiling)
- ✅ **With padding**: Training took ~10 minutes (6x speedup!)

The trade-off is increased memory usage from padding, but on TPU v5litepod-8's 32GB HBM per chip, this isn't a constraint for a 270M model.

**Other TPU-specific settings**:
- `dataloader_drop_last=True`: Ensures all batches have the same size
- `gradient_checkpointing=False`: Standard checkpointing doesn't work on TPU; use `xla_fsdp_grad_ckpt` instead

### Training Execution

With all TPU optimizations in place, initialize the trainer and start training:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=sft_config,
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
```

On a TPU v5litepod-8, training completes in approximately 10 minutes. At the January 2026 on-demand pricing of $2.40/hour, the total training cost comes to around $0.50 — less than a coffee.

Once training finished, I uploaded the adapter weights to Hugging Face Hub as [tengomucho/functiongemma-fitness-coach](https://huggingface.co/tengomucho/functiongemma-fitness-coach). Now anyone can use the fine-tuned model without retraining.

## Building the Fitness Coach Chat Interface

With the fine-tuned model ready, the final step was creating an interactive chatbot. This turned out to be surprisingly straightforward—using FunctionGemma's examples as a starting point, I built a working CLI in less than 200 lines of code.

```console
$ fitnesscoach chat
🏃 Setting up fitness coach with model tengomucho/functiongemma-fitness-coach...
Loading weights: 100%|█| 236/236 [00:00<00:00, 342.36it/s, Materializing param=model.norm.weight]
Loading weights: 100%|█| 144/144 [00:00<00:00, 4091.48it/s, Materializing param=model.layers.17.s

Coach: Hi! How can I help you today? You can ask me questions about your fitness data from today.
You: Tell me the steps walked today
Coach: The number of steps walked today was 1256.
You: What's my energy level?
Coach: My body battery level is 69.
```

### How the Chat Loop Works

The implementation follows a simple pattern: load the model, enter a chat loop, generate a tool call, parse and execute it, then generate a natural language response.

Here's the first generation step where the model decides which function to call:

```python
out = self.model.generate(
    **inputs.to(self.model.device),
    pad_token_id=self.processor.eos_token_id,
    max_new_tokens=128
)

generated_tokens = out[0][len(inputs["input_ids"][0]):]
output = self.processor.decode(generated_tokens, skip_special_tokens=True)
# output can look like this:
# <start_function_call>call:get_body_battery_level{}<end_function_call>
```

The model outputs a structured function call string that's easy to parse with a regex pattern:

```python
function_call = [{
    "name": name,
    "arguments": {
        k: cast((v1 or v2).strip())
        for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
    }
} for name, args in re.findall(r"<start_function_call>call[: ](\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]
```

With the function name and arguments parsed, we can execute the actual function. I maintain a dictionary mapping function names to callable objects, making it straightforward to invoke the right tool and capture its result:

```python
result = self.tools_dict[tool_calls[0]['name']](**tool_calls[0]['arguments'])

# Generate final answer with tool result
messages.append({
    "role": "assistant",
    "tool_calls": [{"type": "function", "function": call} for call in tool_calls]
})
# FunctionGemma expects tool response format with 'name' and 'content'
messages.append({
    "role": "tool",
    "name": tool_calls[0]['name'],
    "content": str(result)
})
```

This builds up the complete conversation history, including the tool execution result. We can now pass this context back to the model for a second generation—this time to produce a natural language answer:

```json
[
    {'role': 'developer', 'content': 'You are a model that can do function calling with the following functions'},
    {'role': 'user', 'content': "What's my energy level?"},
    {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_body_battery_level', 'arguments': {}}}]},
    {'role': 'tool', 'name': 'get_body_battery_level', 'content': '69'},
]
```

A final call to `model.generate` with this complete context produces the natural language answer: "My body battery level is 69."

## Measuring the Impact: Base Model vs. Fine-Tuned

The real question: did the fine-tuning actually help? To find out, I added a `--model-id` flag to the chat CLI so I could swap between the fine-tuned model and the base `google/functiongemma-270m-it`.

Even with just 213 training examples and 10 minutes of training, the improvements were clear. The most striking difference appeared when asking questions in natural language that don't exactly match the function names—cases where the base model would hallucinate (invent non-existent functions).

For example, asking the base model to "display my sleep metrics" triggered a hallucination:

```
You: Please display my sleep metrics
ValueError: Coach tried to call nonexisting tool: get_sleep_metrics
```

The base model invented `get_sleep_metrics`, a function that doesn't exist. When I asked the fine-tuned model the same question, it correctly called `get_sleeping_minutes` and provided the right answer.

This demonstrates that even minimal fine-tuning teaches the model which functions actually exist and when to use them. The fine-tuned model learned to map natural language variations to the correct function names, significantly reducing hallucinations.

With validation complete, let's reflect on what this project teaches us about TPU fine-tuning and small model deployment.

## Conclusion: Key Takeaways from Building a TPU-Powered Fitness Coach

This project started as an experiment with FunctionGemma to understand how small, efficient models could power function-calling applications. The results exceeded expectations: a 270M parameter model fine-tuned for $0.50 in 10 minutes produces reliable, interpretable function calls for a practical application.

The biggest revelation was discovering that **TPU fine-tuning is both faster and cheaper than GPU alternatives**—but only if you understand the hardware constraints. The single most important lesson: TPUs demand static tensor shapes. Without `pad_to_multiple_of=max_length`, training that should take 10 minutes stretches to an hour as the TPU constantly recompiles its computation graph.

### Where to Go From Here

This project demonstrates the viability of TPU fine-tuning for small models, but there's room for improvement:

**Multi-turn conversations**: The 270M model struggles with extended dialogues. This likely stems from the model's size or limited multi-turn training data. Promising alternatives include [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (slightly larger but still edge-deployable) or [tiiuae/Falcon-H1-Tiny-Tool-Calling-90M](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) (even smaller with specialized tool-calling training). The same TPU fine-tuning approach should work for either.

**Richer API design**: The current functions return simple values (integers, tuples). Adding functions that accept parameters or return structured data would test the model's ability to handle more complex tool interactions. Better docstrings and more descriptive function names could also improve the model's function selection accuracy.

**Dataset expansion**: With only 213 examples, the improvement over the base model is noticeable but modest. Scaling to 1,000+ examples covering edge cases, multi-step reasoning, and error handling would likely produce a significantly more robust model. The good news: even at 10x the dataset size, training would still cost under $5 and complete in under an hour.

---

The democratization of AI doesn't require massive GPU clusters or enterprise budgets. With TPUs, you can fine-tune production-ready function-calling models for the cost of a coffee and the time it takes to drink it. Whether you're building fitness coaches, home automation assistants, or domain-specific tools, the path to custom AI has never been more accessible.

All code, datasets, and models are available on [GitHub](https://github.com/tengomucho/fitnesscoach) and [Hugging Face](https://huggingface.co/tengomucho/functiongemma-fitness-coach). Try it yourself and see what you can build in 10 minutes.
