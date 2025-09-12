"""SVAMP prompt for mathematical reasoning tasks."""

from typing import Any, Dict


def get_svamp_prompt(
    prompt_data: Dict[str, Any], behavioral_calibration_prompt: str = ""
) -> str:
    """
    Create a zero-shot Chain of Thought prompt for SVAMP mathematical reasoning.

    Based on research showing that "Let's think step by step" significantly improves
    mathematical reasoning performance on mathematical benchmarks.

    Args:
        prompt_data: Dictionary containing the problem text
        behavioral_calibration_prompt: Additional behavioral calibration instructions

    Returns:
        Formatted prompt with CoT instruction
    """
    problem = prompt_data.get("question_concat", "")

    base_prompt = f"""Let's think step by step.

{problem}

Please work through this problem step by step, showing your reasoning and calculations.

After you've worked through the problem completely, provide your final answer in the JSON format specified below.
"""
    return base_prompt + behavioral_calibration_prompt
