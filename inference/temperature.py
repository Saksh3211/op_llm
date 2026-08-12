"""
Adaptive temperature by task type (spec section 11).

Classifies the prompt with cheap heuristics and suggests a temperature:
low temperature for math/code (want precise, deterministic answers),
higher temperature for open-ended/conversational/creative prompts (want
variety). A manual override always wins over the adaptive guess, so you
can A/B test specific values.
"""

import re

TASK_TEMPERATURES = {
    "math": 0.3,
    "code": 0.4,
    "factual": 0.5,
    "conversational": 0.8,
    "creative": 1.0,
}

DEFAULT_TASK = "conversational"

_MATH_PATTERN = re.compile(r"[\d]+\s*[\+\-\*/=]\s*[\d]+|\bsolve\b|\bcalculate\b|\bequation\b", re.IGNORECASE)
_CODE_PATTERN = re.compile(
    r"\bdef\b|\bfunction\b|\bclass\b|\bimport\b|```|\bcode\b|\bwrite a program\b|\bdebug\b|\bfix this\b",
    re.IGNORECASE,
)
_FACTUAL_PATTERN = re.compile(
    r"^\s*(what|who|when|where|which)\b.*\?\s*$|\bcapital of\b|\bdefine\b", re.IGNORECASE
)
_CREATIVE_PATTERN = re.compile(
    r"\bstory\b|\bpoem\b|\bimagine\b|\bwrite a\b.*\b(story|poem|song)\b|\bonce upon a time\b",
    re.IGNORECASE,
)


def classify_task(prompt: str) -> str:
    """Very cheap keyword/pattern based classifier. Good enough to pick a
    reasonable default temperature; not meant to be precise."""
    if _MATH_PATTERN.search(prompt):
        return "math"
    if _CODE_PATTERN.search(prompt):
        return "code"
    if _CREATIVE_PATTERN.search(prompt):
        return "creative"
    if _FACTUAL_PATTERN.search(prompt):
        return "factual"
    return DEFAULT_TASK


def suggest_temperature(prompt: str) -> tuple[float, str]:
    """Returns (temperature, task_label) for the given prompt."""
    task = classify_task(prompt)
    return TASK_TEMPERATURES[task], task


class TemperatureController:
    """Wraps adaptive suggestion with an optional manual override, so you
    can test fixed temperature values instead of the adaptive guess."""

    def __init__(self):
        self.override: float | None = None

    def set_override(self, value: float | None):
        self.override = value

    def get(self, prompt: str) -> tuple[float, str]:
        """Returns (temperature, source) where source is 'manual' or the
        task label the adaptive guess used."""
        if self.override is not None:
            return self.override, "manual"
        temp, task = suggest_temperature(prompt)
        return temp, task
