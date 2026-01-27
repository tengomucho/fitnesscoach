#!/usr/bin/env python3
"""
Generate a training dataset for fine-tuning FunctionGemma on Garmin fitness APIs.

This script creates synthetic examples of user queries and corresponding function calls
for the following fitness tracking functions:
- get_steps
- get_daily_step_goal
- get_step_goal_progress
- get_sleeping_minutes
- get_active_minutes
- get_heart_rate
- get_body_battery_level

NOTE: Empty dicts are replaced with "empty_dict_placeholder" to avoid Parquet serialization issues.
To restore empty dicts when loading the dataset, use the restore_empty_dicts() function below.

Example usage after loading from HuggingFace:
    from datasets import load_dataset
    dataset = load_dataset('your-username/fitness-coach-function-calling')
    dataset = dataset.map(restore_empty_dicts)
"""

import json
import os
import shutil
from typing import Any

import typer
from datasets import Dataset
from rich import print
from transformers.utils import get_json_schema

# Import the actual functions from fitnesscoach.provider
from fitnesscoach.provider import (
    get_active_minutes,
    get_body_battery_level,
    get_daily_step_goal,
    get_heart_rate,
    get_sleeping_minutes,
    get_step_goal_progress,
    get_steps,
)


def generate_function_definitions() -> list[dict[str, Any]]:
    """Generate function definitions from actual provider functions using transformers.utils.get_json_schema."""
    functions = [
        get_steps,
        get_daily_step_goal,
        get_step_goal_progress,
        get_sleeping_minutes,
        get_active_minutes,
        get_heart_rate,
        get_body_battery_level,
    ]

    function_definitions = []
    for func in functions:
        schema = get_json_schema(func)
        # get_json_schema returns a dict with "type": "function" and "function": {...}
        # We need to extract just the inner function schema
        if isinstance(schema, dict) and "function" in schema:
            schema = schema["function"]

        # Add empty list to required
        schema["parameters"]["required"] = []

        function_definitions.append({
            "type": "function",
            "function": schema
        })

    return function_definitions


# Generate function definitions automatically from provider functions
FUNCTION_DEFINITIONS = generate_function_definitions()


def create_developer_message() -> str:
    """Create the developer message that describes the assistant's role.

    Note that we use default model message from FunctionGemma, see:
    https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf
    """
    return "You are a model that can do function calling with the following functions"


def generate_training_examples() -> list[dict[str, Any]]:
    """Generate synthetic training examples for function calling."""

    examples = []

    # Examples for get_steps - expanded with variations
    examples.extend(
        [
            # Basic queries
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
            # Informal variations
            {"user_query": "yo how many steps today", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "steps?", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "step count pls", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "lemme see my steps", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "how many steps i got", "function_call": {"name": "get_steps", "arguments": {}}},
            # Formal variations
            {
                "user_query": "Could you please provide my step count for today?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {
                "user_query": "I would like to know my daily step count",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {"user_query": "Please display my step count", "function_call": {"name": "get_steps", "arguments": {}}},
            # Different phrasings
            {"user_query": "How many footsteps today?", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Total steps walked", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "My walking steps", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Number of steps taken", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Show steps", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Get my steps", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Check steps", "function_call": {"name": "get_steps", "arguments": {}}},
            # Contextual
            {
                "user_query": "I went for a walk, how many steps did I do?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {
                "user_query": "After my morning jog, what's my step count?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {
                "user_query": "I've been walking all day, show me my steps",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {
                "user_query": "Just got back from a hike, steps?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            # With typos/variations
            {"user_query": "how many stesp today", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "whats my setp count", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "show me stpes", "function_call": {"name": "get_steps", "arguments": {}}},
        ]
    )

    # Examples for get_daily_step_goal - expanded
    examples.extend(
        [
            # Basic queries
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
            # Informal
            {"user_query": "step goal?", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {"user_query": "whats my target steps", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {
                "user_query": "how many steps should i do",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {"user_query": "daily goal pls", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            # Formal
            {
                "user_query": "Could you please tell me my daily step goal?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "I would like to know my step target for today",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "What is my configured daily step goal?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            # Different phrasings
            {"user_query": "My step objective", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {"user_query": "Show my step goal", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {"user_query": "What's my step quota?", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {"user_query": "Daily step target", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {
                "user_query": "How many steps is my goal?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "Target steps for today",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            # Contextual
            {
                "user_query": "I want to know if I'm on track, what's my goal?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "What should I aim for in steps today?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            {
                "user_query": "Trying to plan my day, what's my step goal?",
                "function_call": {"name": "get_daily_step_goal", "arguments": {}},
            },
            # Typos
            {"user_query": "whats my setp goal", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
            {"user_query": "daily step gaol", "function_call": {"name": "get_daily_step_goal", "arguments": {}}},
        ]
    )

    # Examples for get_step_goal_progress - expanded
    examples.extend(
        [
            # Basic queries
            {
                "user_query": "How close am I to my step goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "What's my progress towards my daily goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Am I close to reaching my step target?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Show me my goal progress",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Have I hit my step goal yet?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "What percentage of my step goal have I completed?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            # Informal
            {"user_query": "am i close to my goal", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "how far from goal", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "did i hit my goal", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "goal progress?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "how much left", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "percentage done", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            # Formal
            {
                "user_query": "Could you please show me my progress towards my daily step goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "I would like to know my goal completion percentage",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Please display my step goal progress",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            # Different phrasings
            {
                "user_query": "How much of my goal have I completed?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "What's my step goal completion rate?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Check if I reached my goal",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {"user_query": "Did I make my target?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "Am I on track?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "Show progress", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "How far along am I?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            # Contextual
            {
                "user_query": "I've been walking a lot, did I reach my goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Just finished my workout, how close to goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Before bed, want to check if I hit my goal",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Is it worth going for another walk to hit my goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {
                "user_query": "Halfway through the day, am I on pace?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            # Typos
            {"user_query": "goal progres", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "how close to gaol", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
        ]
    )

    # Examples for get_sleeping_minutes - expanded
    examples.extend(
        [
            # Basic queries
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
            # Informal
            {
                "user_query": "how much sleep did i get",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {"user_query": "sleep time?", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {"user_query": "how long did i sleep", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {"user_query": "my sleep", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {"user_query": "sleep hours pls", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            # Formal
            {
                "user_query": "Could you please provide my sleep duration?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "I would like to know my total sleep time",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Please display my sleep metrics",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            # Different phrasings
            {"user_query": "Total sleep minutes", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {
                "user_query": "How many hours did I rest?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {"user_query": "Sleep data", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {
                "user_query": "My sleep quality last night",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {"user_query": "Check my sleep", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {"user_query": "Show sleep stats", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {"user_query": "How was my sleep?", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            # Contextual
            {
                "user_query": "I'm feeling tired, did I sleep enough?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Woke up groggy, how much did I sleep?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Just woke up, show me my sleep",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Was my sleep good last night?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "I went to bed early, did I get more sleep?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            # Typos
            {"user_query": "how much slep", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
            {"user_query": "slepe duration", "function_call": {"name": "get_sleeping_minutes", "arguments": {}}},
        ]
    )

    # Examples for get_active_minutes - expanded
    examples.extend(
        [
            # Basic queries
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
            # Informal
            {"user_query": "active time?", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "how active am i", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "activity minutes pls", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "show my activity", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "how much activity", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            # Formal
            {
                "user_query": "Could you please show me my active minutes?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "I would like to know my total activity time",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Please display my activity metrics",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            # Different phrasings
            {"user_query": "Total active time", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "My exercise minutes", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {
                "user_query": "How much exercise today?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {"user_query": "Activity level", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "Check my activity", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "How long was I active?", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "Minutes of movement", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            # Contextual
            {
                "user_query": "Had a busy day, was I active enough?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Just finished gym, what's my total active time?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "I want to know if I've been active enough today",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Been sitting all day, how active was I?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "After my workout session, show me active minutes",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            # Typos
            {"user_query": "activ minutes", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "activity minuts", "function_call": {"name": "get_active_minutes", "arguments": {}}},
        ]
    )

    # Examples for get_heart_rate - expanded
    examples.extend(
        [
            # Basic queries
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
            # Informal
            {"user_query": "heart rate?", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "hr today", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "show hr", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "my heart rate", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "whats my hr", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            # Formal
            {
                "user_query": "Could you please provide my heart rate data?",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "I would like to know my heart rate range",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "Please display my cardiovascular metrics",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            # Different phrasings
            {"user_query": "Min and max HR", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "Heart rate range today", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "Check my pulse", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "Show me my BPM", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "Beats per minute", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "Cardiac data", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "My pulse today", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            # Contextual
            {
                "user_query": "Just finished cardio, what was my heart rate?",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "Check my cardiovascular activity from today",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "After my intense workout, show me my HR",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "Was my heart rate elevated today?",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {
                "user_query": "I want to see how hard I worked out, show HR",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            # Typos
            {"user_query": "hart rate", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {"user_query": "heart rat", "function_call": {"name": "get_heart_rate", "arguments": {}}},
        ]
    )

    # Examples for get_body_battery_level - expanded
    examples.extend(
        [
            # Basic queries
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
            # Informal
            {"user_query": "body battery?", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "energy level pls", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "how charged am i", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "my energy", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "battery level", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            # Formal
            {
                "user_query": "Could you please show me my body battery level?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "I would like to know my current energy reserves",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Please display my body battery metrics",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            # Different phrasings
            {
                "user_query": "Check my energy reserves",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {"user_query": "How tired am I?", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "Energy status", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "My charge level", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "How drained am I?", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "Show my battery", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "Stamina level", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            # Contextual
            {
                "user_query": "I'm feeling tired, can you check my body battery?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Should I take a nap? Check my energy",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Feeling drained, what's my body battery?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Do I have enough energy for a workout?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Just woke up from a nap, am I recharged?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Thinking of going to the gym, check my battery",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            # Typos
            {"user_query": "body baterry", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
            {"user_query": "enrgy level", "function_call": {"name": "get_body_battery_level", "arguments": {}}},
        ]
    )

    # Additional contextual and conversational examples
    examples.extend(
        [
            {"user_query": "Give me a fitness update", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "How am I doing today?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "Show me my stats", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Quick fitness check", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Health summary", "function_call": {"name": "get_steps", "arguments": {}}},
            {"user_query": "Am I doing well today?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {
                "user_query": "How's my day going fitness wise?",
                "function_call": {"name": "get_steps", "arguments": {}},
            },
            {"user_query": "Show me my movement today", "function_call": {"name": "get_steps", "arguments": {}}},
            {
                "user_query": "I need to know my fitness level",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Can you check my activity?",
                "function_call": {"name": "get_active_minutes", "arguments": {}},
            },
            {
                "user_query": "Tell me about my rest",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "How well did I recover?",
                "function_call": {"name": "get_sleeping_minutes", "arguments": {}},
            },
            {
                "user_query": "Am I recovered from yesterday?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Can I work out now?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {
                "user_query": "Should I rest or go exercise?",
                "function_call": {"name": "get_body_battery_level", "arguments": {}},
            },
            {"user_query": "How intense was my day?", "function_call": {"name": "get_heart_rate", "arguments": {}}},
            {
                "user_query": "Did I push myself hard today?",
                "function_call": {"name": "get_heart_rate", "arguments": {}},
            },
            {"user_query": "Was I lazy today?", "function_call": {"name": "get_active_minutes", "arguments": {}}},
            {"user_query": "Did I move around enough?", "function_call": {"name": "get_steps", "arguments": {}}},
            {
                "user_query": "Should I walk more today?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {"user_query": "Need more steps?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {"user_query": "Am I behind on my goal?", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {
                "user_query": "How many more steps do I need?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
            {"user_query": "Steps remaining", "function_call": {"name": "get_step_goal_progress", "arguments": {}}},
            {
                "user_query": "What's left to reach my goal?",
                "function_call": {"name": "get_step_goal_progress", "arguments": {}},
            },
        ]
    )

    return examples


def format_for_function_calling(examples: list[dict[str, Any]]) -> dict[str, list]:
    """Format examples for FunctionGemma training.

    Creates a dataset with the following columns:
    - messages: Array of message objects with role and content
    - tools: Available function definitions
    """

    developer_message = create_developer_message()

    formatted_data = {
        "messages": [],
        "tools": [],
    }

    for example in examples:
        # Create messages array following FunctionGemma format
        function_call = example["function_call"]

        messages = [
            {
                "role": "developer",
                "content": developer_message
            },
            {
                "role": "user",
                "content": example["user_query"]
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": function_call["name"],
                            "arguments": function_call["arguments"]
                        }
                    }
                ]
            }
        ]

        formatted_data["messages"].append(json.dumps(messages))
        formatted_data["tools"].append(json.dumps(FUNCTION_DEFINITIONS))

    return formatted_data


def cleanup_generated_files():
    """Remove all generated dataset files and directories."""
    dirs_to_remove = ["dataset/fitness_coach_function_calling"]

    print("Cleaning up generated files...")

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

    print("\nFormatting examples for FunctionGemma...")
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
        example = dataset[i]
        print(example)

    # Save to disk in HuggingFace Dataset format
    output_dir = "dataset/fitness_coach_function_calling"

    print(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    print("Saved dataset to HuggingFace Dataset format")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total examples: {len(dataset)}")

    # Count examples per function
    from collections import Counter

    function_counts = Counter()
    for messages in dataset["messages"]:
        messages = json.loads(messages)
        # Extract function name from assistant's tool_calls
        for msg in messages:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    function_counts[tool_call["function"]["name"]] += 1

    print("\nExamples per function:")
    for func_name, count in sorted(function_counts.items()):
        print(f"  {func_name}: {count}")

    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print("=" * 50)
    print("\nFiles created in:")
    print(f"  {output_dir}")
    print("\nYou can now upload the dataset to HuggingFace Hub!")
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
