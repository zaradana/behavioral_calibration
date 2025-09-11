from typing import Dict

# reproduced from https://github.com/idavidrein/gpqa/blob/a99e185c4fed60a4181040415b63771b2941b998/baselines/utils.py


def base_prompt(example: Dict[str, str]) -> str:
    """Creates a zero-shot prompt given a single example. Uses the prompt format from this paper on Scalable Oversight:
    https://arxiv.org/abs/2211.03540"""

    prompt = f"What is the correct answer to this question: {example['question']}"
    prompt += f"\n\nChoices:\n(A) {example['choice1']}\n(B) {example['choice2']}\n(C) {example['choice3']}\n(D) {example['choice4']}"
    return prompt


def zero_shot_prompt(example: Dict[str, str]) -> str:
    """Creates a zero-shot prompt given a single example. Uses the prompt format from this paper on Scalable Oversight:
    https://arxiv.org/abs/2211.03540"""
    prompt = base_prompt(example)
    prompt += '\n\nFormat your response as follows: "The correct answer is (insert answer here)"'
    return prompt


def get_gpqa_prompt(
    prompt_data: Dict[str, str], behavioral_calibration_prompt: str = ""
) -> str:
    """Creates a prompt given processed prompt data."""
    return zero_shot_prompt(prompt_data) + behavioral_calibration_prompt
