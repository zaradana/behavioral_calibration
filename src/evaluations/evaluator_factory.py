"""Factory for creating benchmark evaluators."""

from typing import Dict, Type

from ..core.schema import BenchmarkConfig
from .base_evaluator import BaseBenchmarkEvaluator
from .gsm8k_evaluator import GSM8KEvaluator
from .mc_evaluator import MCEvaluator
from .proxy_evaluator import ProxyEvaluator
from .svamp_evaluator import SVAMPEvaluator
from .swe_evaluator import SWEEvaluator


class EvaluatorFactory:
    """Factory class for creating appropriate benchmark evaluators."""

    # Registry mapping dataset names to evaluator classes
    _evaluator_registry: Dict[str, Type[BaseBenchmarkEvaluator]] = {
        "proxy_data": ProxyEvaluator,
        "gpqa": MCEvaluator,
        "truthfulqa": MCEvaluator,
        "swe": SWEEvaluator,
        "gsm8k": GSM8KEvaluator,
        "svamp": SVAMPEvaluator,
    }

    @classmethod
    def create_evaluator(
        cls, benchmark_config: BenchmarkConfig
    ) -> BaseBenchmarkEvaluator:
        """
        Create an appropriate evaluator for the given benchmark configuration.

        Args:
            benchmark_config: Configuration for the benchmark

        Returns:
            Evaluator instance for the benchmark type

        Raises:
            ValueError: If no evaluator is found for the benchmark type
        """
        dataset_name = benchmark_config.dataset_name.lower()

        # Try exact match first
        if dataset_name in cls._evaluator_registry:
            evaluator_class = cls._evaluator_registry[dataset_name]
            return evaluator_class(benchmark_config)

        # Try partial matches for SWE-bench variants
        for pattern, evaluator_class in cls._evaluator_registry.items():
            if pattern in dataset_name:
                return evaluator_class(benchmark_config)

        # No evaluator found
        available_types = list(cls._evaluator_registry.keys())
        raise ValueError(
            f"No evaluator found for benchmark '{dataset_name}'. "
            f"Available benchmark types: {available_types}"
        )

    @classmethod
    def get_supported_benchmarks(cls) -> list[str]:
        """Get list of supported benchmark types."""
        return list(cls._evaluator_registry.keys())
