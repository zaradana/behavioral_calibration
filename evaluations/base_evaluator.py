"""Base evaluator interface for benchmark evaluation."""

from abc import ABC, abstractmethod
from typing import List

from schema import BenchmarkConfig, ItemEval


class BaseBenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""

    def __init__(self, benchmark_config: BenchmarkConfig):
        """Initialize evaluator with benchmark configuration."""
        self.benchmark_config = benchmark_config

    @abstractmethod
    async def evaluate_predictions(
        self,
        predictions: List[ItemEval],
        confidence_threshold: float,
        model_name: str,
        filename_id: str,
    ) -> List[ItemEval]:
        """
        Evaluate predictions for this benchmark type.

        Args:
            predictions: List of predictions to evaluate
            confidence_threshold: Threshold for confidence-based decisions
            model_name: Name of the model being evaluated
            filename_id: Unique identifier for this evaluation run

        Returns:
            List of evaluated predictions with correctness and payoff calculated
        """
        pass

    def get_benchmark_name(self) -> str:
        """Get the name of this benchmark."""
        return self.benchmark_config.dataset_name
