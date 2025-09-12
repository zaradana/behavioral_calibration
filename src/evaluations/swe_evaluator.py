"""SWE-bench evaluator."""

import json
import os
from typing import List

from ..core.config import OUTPUT_DIR
from ..core.schema import ItemEval
from ..utils.benchmarks.swe_utils import (
    prepare_swebench_predictions,
    run_swebench_with_docker,
)
from ..utils.evaluation_utils import payoff_value
from .base_evaluator import BaseBenchmarkEvaluator


class SWEEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for SWE-bench tasks."""

    async def evaluate_predictions(
        self,
        predictions: List[ItemEval],
        confidence_threshold: float,
        model_name: str,
        filename_id: str,
    ) -> List[ItemEval]:
        """Evaluate SWE-bench predictions using Docker-based test execution."""

        # Prepare predictions for SWE-bench
        swe_bench_predictions = prepare_swebench_predictions(predictions, model_name)
        predictions_path = (
            f"{OUTPUT_DIR}/{model_name}/{model_name}_patches_{filename_id}.jsonl"
        )
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        with open(predictions_path, "w") as f:
            for pred in swe_bench_predictions:
                f.write(json.dumps(pred) + "\n")

        swebench_report_path = run_swebench_with_docker(
            predictions_path=predictions_path,
            dataset_name=self.benchmark_config.dataset_path,
            split=self.benchmark_config.dataset_split,
            run_id=f"{model_name}_{filename_id}",
        )

        if swebench_report_path:
            swebench_results = json.load(open(swebench_report_path))
            resolved_ids = set(swebench_results["completed_ids"])
        else:
            swebench_results = []
            resolved_ids = set()

        # Evaluate each prediction based on SWE-bench results
        outputs: List[ItemEval] = []
        for pred in predictions:
            is_correct = (
                True
                if (
                    pred.decision == "answer"
                    and pred.evaluation_metadata["instance_id"] in resolved_ids
                )
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
