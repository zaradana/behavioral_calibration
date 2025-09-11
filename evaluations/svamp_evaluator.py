"""SVAMP evaluator for mathematical reasoning with exact match."""

import re
from typing import List

from schema import ItemEval
from utils.evaluation_utils import payoff_value

from .base_evaluator import BaseBenchmarkEvaluator


def extract_svamp_answer(text: str) -> str:
    """
    Extract the final numeric answer from SVAMP model response.
    Since we use JSON format, try to extract from answer field or find numbers in text.

    Args:
        text: Model response text

    Returns:
        Extracted numeric answer as string, or full text if no number found
    """
    # Try to extract any number from the text
    stripped_text = text.strip()

    # Look for numbers in the text
    number_pattern = r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
    numbers = re.findall(number_pattern, stripped_text)
    if numbers:
        # Take the last number found (most likely to be the answer)
        last_number = numbers[-1].replace(",", "")
        # Clean it of symbols
        cleaned_number = re.sub(r'[\$€£¥%°#@&*≈~:,\s\(\)\[\]{}"\'`]', "", last_number)
        if (
            cleaned_number
            and cleaned_number.replace(".", "")
            .replace("-", "")
            .replace("+", "")
            .isdigit()
        ):
            return cleaned_number

    # If no number found, return the original stripped text
    return stripped_text


def normalize_svamp_answer(answer: str) -> str:
    """
    Normalize numeric answer for comparison.

    Args:
        answer: Numeric answer as string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Remove any remaining symbols, commas, and whitespace
    cleaned = re.sub(r'[\$€£¥%°#@&*≈~:,\s\(\)\[\]{}"\'`]', "", answer.strip())

    try:
        # Try to convert to float then back to string to normalize format
        if "." in cleaned:
            normalized = str(float(cleaned))
            # Remove trailing .0 for integers
            if normalized.endswith(".0"):
                normalized = normalized[:-2]
            return normalized
        else:
            return str(int(cleaned))
    except (ValueError, TypeError):
        return cleaned


def evaluate_svamp_answer(predicted_answer: str, correct_answer: str) -> bool:
    """
    Evaluate SVAMP answer using exact match of numeric values.

    Args:
        predicted_answer: Model's predicted answer
        correct_answer: Ground truth answer

    Returns:
        True if answers match exactly, False otherwise
    """
    # Extract numeric answers
    pred_numeric = extract_svamp_answer(predicted_answer)

    # Normalize both answers
    pred_normalized = normalize_svamp_answer(pred_numeric)
    correct_normalized = normalize_svamp_answer(correct_answer)

    return pred_normalized == correct_normalized and pred_normalized != ""


class SVAMPEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for SVAMP mathematical reasoning benchmark."""

    async def evaluate_predictions(
        self,
        predictions: List[ItemEval],
        confidence_threshold: float,
        model_name: str,
        filename_id: str,
    ) -> List[ItemEval]:
        """Evaluate SVAMP predictions using exact match on numeric answers."""
        outputs: List[ItemEval] = []

        for pred in predictions:
            # Get the correct answer from metadata
            correct_answer = pred.evaluation_metadata.get("answer", "")

            # Evaluate using exact match
            is_correct = evaluate_svamp_answer(pred.answer, correct_answer)

            # Calculate payoff
            pay = payoff_value(is_correct, pred.decision, confidence_threshold)

            outputs.append(
                ItemEval(
                    id=pred.id,
                    decision=pred.decision,
                    answer=pred.answer,
                    confidence=pred.confidence,
                    correct=is_correct,
                    payoff_behavioral=pay,
                    evaluation_metadata=pred.evaluation_metadata,
                )
            )

        return outputs
