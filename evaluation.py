import json
import os
from typing import List

from config import OUTPUT_DIR
from schema import BenchmarkConfig, ItemEval
from utils.benchmarks.proxy_utils import correctness_heuristic
from utils.benchmarks.swe_utils import (
    prepare_swebench_predictions,
    run_swebench_with_docker,
)
from utils.evaluation_utils import payoff_value


async def evaluate_model(
    predictions: List[ItemEval],
    benchmark_config: BenchmarkConfig,
    confidence_threshold: float,
    model_name: str,
    filename_id: str,
) -> List[ItemEval]:
    """
    Evaluate a model across different targets with optional SWE-bench integration.

    Args:
        predictions: List of evaluation items
        benchmark_config: Benchmark configuration
        temperature: Temperature for the model
        model_name: Name of the model
        filename_id: ID for predictions and results file names

    Returns:
        List of evaluation results
    """
    # Run behavioral evaluation
    outputs: List[ItemEval] = []

    # Handle different benchmark types based on dataset name
    if benchmark_config.dataset_name == "proxy_data":
        # Use proxy data evaluation with heuristic correctness
        for pred in predictions:
            # Find the corresponding item for expect_substrings
            is_correct = (
                correctness_heuristic(pred.answer, pred.item_id)
                if pred.decision == "answer"
                else False
            )
            pay = payoff_value(is_correct, pred.decision, confidence_threshold)
            outputs.append(
                ItemEval(
                    item_id=pred.item_id,
                    decision=pred.decision,
                    answer=pred.answer,
                    confidence=pred.confidence,
                    correct=is_correct,
                    payoff_behavioral=pay,
                )
            )
    elif "swe" in benchmark_config.dataset_name.lower():
        # Prepare predictions for SWE-bench
        swe_bench_predictions = prepare_swebench_predictions(predictions, model_name)
        predictions_path = (
            f"{OUTPUT_DIR}/{model_name}/{model_name}_patches_{filename_id}.jsonl"
        )
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        with open(predictions_path, "w") as f:
            for pred in swe_bench_predictions:
                f.write(json.dumps(pred) + "\n")

        if predictions:
            swebench_report_path = run_swebench_with_docker(
                predictions_path=predictions_path,
                dataset_name=benchmark_config.dataset_path,
                split=benchmark_config.dataset_split,
                run_id=f"{model_name}_{filename_id}",
            )
            if swebench_report_path:
                swebench_results = json.load(open(swebench_report_path))
                resolved_ids = set(swebench_results["completed_ids"])
            else:
                swebench_results = []
                resolved_ids = set()

            # For SWE-bench, we'll need to get correctness from actual evaluation
            # For now, use a mock correctness based on having a non-empty patch
            for pred in predictions:
                is_correct = (
                    True
                    if (pred.decision == "answer" and pred.item_id in resolved_ids)
                    else False
                )
                pay = payoff_value(is_correct, pred.decision, confidence_threshold)
                outputs.append(
                    ItemEval(
                        item_id=pred.item_id,
                        decision=pred.decision,
                        answer=pred.answer,
                        confidence=pred.confidence,
                        correct=is_correct,
                        payoff_behavioral=pay,
                    )
                )

    return outputs
