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

The first step was to define some very simple APIs that could be invoked as tool calls from the chat.
I used the very handy `garminconnect` python module to retrieve the data from my Garmin watch, and I defined few functions to retrieve the information in the most simple way. E.g., to retrieve the number of steps walked today I have:

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

Similarly, I defined these functions:

- get_steps
- get_daily_step_goal
- get_step_goal_progress
- get_sleeping_minutes
- get_active_minutes
- get_heart_rate
- get_body_battery_level


To test the API, I created a simple command line interface (CLI) that could be invoked to retrieve some of the data:

```bash
$ fitnesscoach summary
Steps: 2400 (31.41%)
Daily Step Goal: 7640
Active time: 1h:3m
Sleeping time: 7h:13m
```

## Creating a Dataset for Fine Tuning

When using the FunctionGemma model, the suggested way of preparing the messages in the format expected by the model is to use the following structure:

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

The inputs can be then used with the `model.generate` call, that will provide a tool calling message. The tool call can be parsed so to obtain a message dialog that can be provided to the model to obtain the final response:

```json
[
    {'role': 'developer', 'content': 'You are a model that can do function calling with the following functions'},
    {'role': 'user', 'content': 'tell me the steps walked'},
    {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_steps', 'arguments': {}}}]},
    {'role': 'tool', 'name': 'get_steps', 'content': '2400'},
    {'role': 'assistant', 'content': 'The number of steps walked was 2400.'}
]
```

For a complete example on how to do this, you can check the FunctionGemma [model card](https://huggingface.co/google/functiongemma-270m-it).
What I wanted to do is to create a synthetic dataset that matches the calls.
I ended up writing a script that created a dataset with 213 examples of possible conversations with the fitness coach and the tool call that should be called. E.g.:

```json
{"user_query": "My walking steps", "function_call": {"name": "get_steps", "arguments": {}}},
{"user_query": "Number of steps taken", "function_call": {"name": "get_steps", "arguments": {}}},
{"user_query": "Show steps", "function_call": {"name": "get_steps", "arguments": {}}},
{"user_query": "Get my steps", "function_call": {"name": "get_steps", "arguments": {}}},
```

Once these have been defined, I used `get_json_schema` from `transformers.utils` to automatically parse the API metadata and generate the correct JSON tool definition that can be passed over to the chat template.
With these, it was possible to create the simple [fitness coach function calling](https://huggingface.co/datasets/tengomucho/fitness-coach-function-calling) dataset, that I then shared on the Hugging Face hub.

## Fine Tuning the Model on TPU

While inference on FunctionGemma can be done on a common PC, fine-tuning requires significant compute and memory. This is where Google's TPUs shine—but they require specific optimizations to achieve peak performance.

Without proper configuration, TPU training can actually be **slower than GPU** due to repeated graph compilation. This section covers the three critical optimizations that reduced training time from ~1 hour to ~10 minutes.

### TPU Optimization #1: SPMD Initialization

First, enable Single Program Multiple Data (SPMD) mode **before** loading the model. This is required for FSDPv2 on TPU:

```python
import torch_xla.runtime as xr
xr.use_spmd()  # Must call before model initialization
```

⚠️ **Important**: Call `use_spmd()` before any model operations. Calling it after model initialization will cause errors.

### Loading the Model

Load the model using the standard `transformers` API:

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

Configure LoRA for parameter-efficient fine-tuning:

```python
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

### TPU Optimization #3: Static Tensor Shapes (Critical!)

This is the **most important optimization** for TPU performance:

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

On a TPU v5litepod-8 the training takes around 10 minutes. As of January 2026, the price for this setup is $2.40/hour with the on-demand pricing. The total cost for the training should be around $0.50.
Once the model adapter has been trained, it can be uploaded to the Hugging Face hub. My version is available as [tengomucho/functiongemma-fitness-coach](https://huggingface.co/tengomucho/functiongemma-fitness-coach).

## Fitness Coach Chat

With a new fine-tuned model, it was now time to create the actual fitness coach chatbot. Doing it is rather simple, I used the code sample on the FunctionGemma as a starting point and crafted the chatbot in less than 200 lines of code. You can launch it with a command from the CLI.

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

The demo is quite simple, it loads the model and the tokenizer and then it enters into a chat loop that asks for a prompt, it generates a tool call, it parses the tool call and runs it, and finally it generates a final answer based on the tool call result.

At the beginning of each chat step, the FunctionGemma messages template is initialized, then we add the question, apply the template and the obtain the tool call output:

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

The output can be easily parsed to call a registered tool:

```python
function_call = [{
    "name": name,
    "arguments": {
        k: cast((v1 or v2).strip())
        for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
    }
} for name, args in re.findall(r"<start_function_call>call[: ](\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]
```

We registered a dictionary of the allowed tools, so we just find out if the tool call request matches a name from the tool call dictionary keys, and then make the call. We then add the call result into the messages:

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

This will generate a dictionary with the complete context that we can pass as argument to the `apply_chat_template` function:

```json
[
    {'role': 'developer', 'content': 'You are a model that can do function calling with the following functions'},
    {'role': 'user', 'content': "What's my energy level?"},
    {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_body_battery_level', 'arguments': {}}}]},
    {'role': 'tool', 'name': 'get_body_battery_level', 'content': '69'},
]
```

Once we call the final `model.generate` with this context, we obtain the final answer.

## Compare with the Base Model

While the default model chosen for the chat demo is the fine-tuned `tengomucho/functiongemma-fitness-coach`, I added the possibility to choose another model, so I could compare the results with the base model, `google/functiongemma-270m-it`.

Even with the short fine-tuning based on the small dataset created, I could see some improvements over the base model. One of these is when asking questions in natural language that could lead the base model to hallucinated tool calls (when the model invents non-existent functions). For example, when the base model was asked to report the sleep metrics, it incorrectly tried to call a function that does not exist:

```
You: Please display my sleep metrics
ValueError: Coach tried to call nonexisting tool: get_sleep_metrics
```

When using the same prompt on the fine-tuned model, it calls the expected tool, i.e.: `get_sleeping_minutes` and answers correctly.
Given this, I can conclude that fine-tuning FunctionGemma leads to more relevant and precise tool calls.

## Conclusion: What I Learned when Creating a Virtual Fitness Coach

I started this project to experiment with FunctionGemma and try to understand how it could be used on simple applications. I quickly figured out that the model is a great tiny model, great to generate interpretable tool calls after a given prompt. It was quickly clear to me the value of fine-tuning this model, and while some resources are available on how to do this on a GPU, I was happy to try to do that on TPU. I found out that fine-tuning on Google's TPU is rather easy and fast, in particular if we are aware of the hardware implementation details, in particular keeping the same shape all along the training loop. This can be achieved easily by setting the parameters to the trainer as explained in the above section.

Here's a list of things that could be improved in this project:

- I found the model does not seem to work well with multi-turn conversations. This might be due to the fact that the model is rather small, or perhaps that is has not been trained too much to achieve great performance on this. Another option would be to replace it with a slightly bigger model, for example [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), or with a more recent one, like the more recent [tiiuae/Falcon-H1-Tiny-Tool-Calling-90M](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M). I haven't tried them on this project, but I am pretty sure that fine-tuning them could lead to performant results too.
- The API might be more complete and more "LLM friendly", so to make it easier for the model to figure out which function should be called and better interpret results.
- As mentioned before, the generated dataset is not very extensive, so the improvement over the base model is visible but marginal. A larger dataset can lead to better fine-tuning results.
