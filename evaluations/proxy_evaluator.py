"""Proxy benchmark evaluator."""

from typing import List
from data_loader import get_data
from benchmarks import get_benchmark_config
from schema import ItemEval, BenchmarkConfig
from utils.evaluation_utils import payoff_value
from .base_evaluator import BaseBenchmarkEvaluator

class ProxyEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for proxy benchmark tasks."""
    
    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__(benchmark_config)
        self.raw_dataset = get_data(benchmark_config=benchmark_config)
    
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
                self._correctness_heuristic(pred.answer, pred.evaluation_metadata["id"])
                if pred.decision == "answer"
                else False
            )
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

    def _correctness_heuristic(self, answer: str, item_id: str) -> bool:
        """Simple heuristic to check if an answer contains expected substrings."""
        a = (answer or "").lower()
        expect_substrings = next(item["expect_substrings"] for item in self.raw_dataset if item["id"] == item_id)
        return all(s in a for s in (s.lower() for s in expect_substrings))
