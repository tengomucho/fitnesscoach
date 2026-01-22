import re
from typing import Any

from rich import print
from transformers import AutoModelForCausalLM, AutoProcessor

from .provider import (
    get_active_minutes,
    get_body_battery_level,
    get_daily_step_goal,
    get_goal_progress,
    get_heart_rate,
    get_sleeping_minutes,
    get_steps,
)


MODEL_ID = "google/functiongemma-270m-it"


TOOLS = [
        get_steps,
        get_daily_step_goal,
        get_goal_progress,
        get_sleeping_minutes,
        get_active_minutes,
        get_heart_rate,
        get_body_battery_level,
    ]

def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """
    Extract tool calls from a text string.
    Based on the code from FunctionGemma documentation:
    https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma


    Args:
        text: The text string to extract tool calls from.

    Returns:
        A list of tool calls.
    """
    def cast(v):
        try:
            return int(v)
        except: # noqa: E722
            try:
                return float(v)
            except: # noqa: E722
                return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]



class FitnessCoachChat:
    def __init__(self, verbose: bool = False):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
        self.tools = TOOLS
        self.tools_dict = {tool.__name__: tool for tool in self.tools}
        self.messages = [
            {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        ]
        self.verbose = verbose

    def log(self, message: str):
        if self.verbose:
            print(message)

    def ask(self, question: str) -> str:
        """
        Ask a question to the model.
        """
        self.messages.append({"role": "user", "content": question})
        return self._chat_step()


    def _format_inputs(self):
        """
        Get the inputs for the model.
        """
        inputs = self.processor.apply_chat_template(
            self.messages,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        return inputs

    def _chat_step(self) -> str:
        """
        Chat step implementation that manually handles the full FunctionGemma flow.
        This demonstrates the complete cycle: question -> tool call -> tool result -> answer.
        """
        inputs = self._format_inputs()

        out = self.model.generate(
            **inputs.to(self.model.device),
            pad_token_id=self.processor.eos_token_id,
            max_new_tokens=128
        )

        generated_tokens = out[0][len(inputs["input_ids"][0]):]
        output = self.processor.decode(generated_tokens, skip_special_tokens=True)
        self.log(f"🤖 Model output: {output}")

        # Parse and execute tool calls
        tool_calls = extract_tool_calls(output)
        if not tool_calls:
            return output

        # Run the tool
        self.log(f"\n🔧 Executing tool: {tool_calls[0]['name']}")
        result = self.tools_dict[tool_calls[0]['name']](**tool_calls[0]['arguments'])
        self.log(f"📊 Tool result: {result}")

        # Generate final answer with tool result
        self.messages.append({
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": call} for call in tool_calls]
        })
        # FunctionGemma expects tool response format with 'name' and 'content'
        self.messages.append({
            "role": "tool",
            "name": tool_calls[0]['name'],
            "content": str(result)
        })

        inputs = self._format_inputs()

        out = self.model.generate(
            **inputs.to(self.model.device),
            pad_token_id=self.processor.eos_token_id,
            max_new_tokens=128
        )

        generated_tokens = out[0][len(inputs["input_ids"][0]):]
        final_output = self.processor.decode(generated_tokens, skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": final_output})
        return final_output


def _one_question_chat(coach: FitnessCoachChat, question: str) -> str:
    """
    One question chat implementation.
    """
    print(f"💬 Question: {question}\n")
    final_output = coach.ask(question)
    print(f"\n✅ Final answer: {final_output}")
    return final_output


def _chat_loop(coach: FitnessCoachChat):
    """
    Chat loop implementation.
    """
    import typer  # noqa: F401

    print("\nCoach: Hi! How can I help you today? You can ask me questions about your fitness data from today.")
    question = ""
    while question != "exit" and question != "quit":
        question = typer.prompt("You")
        output = coach.ask(question)
        print(f"\nCoach: {output}")

def chat(question: str | None = None, verbose: bool = False):
    """
    Simple chat implementation that manually handles the full FunctionGemma flow.
    This demonstrates the complete cycle: question -> tool call -> tool result -> answer.
    """
    print("🏃 Setting up fitness coach...")

    coach = FitnessCoachChat(verbose=verbose)

    if question:
        return _one_question_chat(coach, question)
    else:
        return _chat_loop(coach)
