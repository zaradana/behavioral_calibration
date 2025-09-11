"""GPQA benchmark evaluator."""

import re
from typing import List

from schema import ItemEval
from utils.evaluation_utils import payoff_value

from .base_evaluator import BaseBenchmarkEvaluator


def extract_gpqa_answer(response_text: str) -> str:
    """Extract the answer choice (A, B, C, D) from the model response."""
    if not response_text:
        return ""

    # Look for patterns like "The correct answer is (A)" or just "(A)" or "A"
    patterns = [
        r"answer is \(([ABCD])\)",
        r"\(([ABCD])\)",
        r"^([ABCD])$",  # Just a single letter
        r"answer is ([ABCD])",  # Without parentheses
    ]

    response_text = response_text.strip().upper()

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If no pattern matched, try to find any single A, B, C, or D
    letters = re.findall(r"[ABCD]", response_text)
    if letters:
        return letters[0]  # Return the first found letter

    return ""  # No answer found


def evaluate_gpqa_answer(predicted_answer: str, correct_index: int) -> bool:
    """Evaluate if the predicted answer matches the correct index."""
    if not predicted_answer:
        return False

    # Convert correct_index (0-3) to letter (A-D)
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct_letter = index_to_letter.get(correct_index, "")

    return predicted_answer.upper() == correct_letter


class GPQAEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for GPQA benchmark tasks."""

    async def evaluate_predictions(
        self,
        predictions: List[ItemEval],
        confidence_threshold: float,
        model_name: str,
        filename_id: str,
    ) -> List[ItemEval]:
        """Evaluate GPQA predictions by extracting and comparing answer choices."""
        outputs: List[ItemEval] = []

        for pred in predictions:
            extracted_answer = extract_gpqa_answer(pred.answer)
            correct_index = pred.evaluation_metadata.get("correct_index")
            is_correct = evaluate_gpqa_answer(extracted_answer, correct_index)
            pay = payoff_value(is_correct, pred.decision, confidence_threshold)
            outputs.append(
                ItemEval(
                    decision=pred.decision,
                    answer=pred.answer,
                    confidence=pred.confidence,
                    correct=is_correct,
                    payoff_behavioral=pay,
                    evaluation_metadata=pred.evaluation_metadata,
                )
            )

        return outputs
