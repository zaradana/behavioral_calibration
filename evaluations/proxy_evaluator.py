"""Proxy benchmark evaluator."""

from typing import List

from schema import ItemEval
from utils.evaluation_utils import payoff_value

from .base_evaluator import BaseBenchmarkEvaluator


class ProxyEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for proxy benchmark tasks."""

    async def evaluate_predictions(
        self,
        predictions: List[ItemEval],
        confidence_threshold: float,
        model_name: str,
        filename_id: str,
    ) -> List[ItemEval]:
        """Evaluate proxy predictions using heuristic correctness check."""
        outputs: List[ItemEval] = []

        for pred in predictions:
            is_correct = (
                self._correctness_heuristic(
                    pred.answer, pred.evaluation_metadata["expect_substrings"]
                )
                if pred.decision == "answer"
                else False
            )
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

    def _correctness_heuristic(self, answer: str, expect_substrings: List[str]) -> bool:
        """Simple heuristic to check if an answer contains expected substrings."""
        a = (answer or "").lower()
        return all(s in a for s in (s.lower() for s in expect_substrings))
