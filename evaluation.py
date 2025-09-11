from typing import List

from evaluations.evaluator_factory import EvaluatorFactory
from schema import BenchmarkConfig, ItemEval


async def evaluate_model(
    predictions: List[ItemEval],
    benchmark_config: BenchmarkConfig,
    confidence_threshold: float,
    model_name: str,
    filename_id: str,
) -> List[ItemEval]:
    """
    Evaluate a model across different benchmarks using the evaluator pattern.

    Args:
        predictions: List of evaluation items
        benchmark_config: Benchmark configuration
        confidence_threshold: Threshold for confidence-based decisions
        model_name: Name of the model
        filename_id: ID for predictions and results file names

    Returns:
        List of evaluation results
    """
    # Create appropriate evaluator for the benchmark type
    evaluator = EvaluatorFactory.create_evaluator(benchmark_config)

    # Delegate evaluation to the specific evaluator
    return await evaluator.evaluate_predictions(
        predictions=predictions,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        filename_id=filename_id,
    )
