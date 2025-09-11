import asyncio
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from agent import BehavioralCalibrationAgent
from config import BENCHMARK, CONFIDENCE_TARGETS, MODELS, OUTPUT_DIR, TEMPERATURE
from data_loader import get_data
from evaluation import evaluate_model
from schema import ConfigSchema, ItemEval
from utils.core_utils import get_logger, save_model_results
from utils.evaluation_utils import confidence_by_correctness, summarize_behavioral
from visualization import plot_overlays

logger = get_logger(__name__)


async def get_predictions(
    data: List[Dict[str, Any]],
    agent: BehavioralCalibrationAgent,
    confidence_threshold: float,
) -> List[ItemEval]:
    """Get predictions for all data items sequentially (to avoid overwhelming the API)."""
    predictions: List[ItemEval] = []
    for item in data:
        obj = await agent.run(item, confidence_threshold)
        predictions.append(
            ItemEval(
                id=obj.id,
                decision=obj.decision,
                answer=obj.answer,
                confidence=obj.confidence,
                evaluation_metadata=obj.evaluation_metadata,
            )
        )
    return predictions


async def process_confidence_threshold(
    data: List[Dict[str, Any]],
    agent: BehavioralCalibrationAgent,
    confidence_threshold: float,
    model_name: str,
    filename_id: str,
) -> tuple[float, List[ItemEval]]:
    """Process a single confidence threshold and return results."""
    start_time = datetime.now()
    logger.info(
        "[%s] Getting predictions for %s with confidence threshold %s",
        start_time.strftime("%H:%M:%S.%f")[:-3],
        model_name,
        confidence_threshold,
    )
    # data = list(data)[:5] # for testing
    predictions: List[ItemEval] = await get_predictions(
        data, agent, confidence_threshold
    )

    eval_start_time = datetime.now()
    logger.info(
        "[%s] Evaluating %s with threshold %s...",
        eval_start_time.strftime("%H:%M:%S.%f")[:-3],
        model_name,
        confidence_threshold,
    )

    # Run evaluation with optional SWE-bench integration
    evaluation_results = await evaluate_model(
        predictions=predictions,
        benchmark_config=BENCHMARK,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        filename_id=filename_id,
    )

    return confidence_threshold, evaluation_results


async def main():
    data = get_data(benchmark_config=BENCHMARK)

    agg_results_by_model: Dict[str, Dict[float, Dict[str, float]]] = {}
    output = []
    filename_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for model_config in MODELS:
        model_name = model_config.model_name

        logger.info(
            "Processing all confidence thresholds concurrently for %s", model_name
        )

        # Add timing to see what's happening
        overall_start = datetime.now()
        logger.info(
            "[%s] Starting all threshold tasks",
            overall_start.strftime("%H:%M:%S.%f")[:-3],
        )

        # Process all confidence thresholds concurrently
        # Create separate agent instances to avoid shared connection/state issues
        tasks = [
            process_confidence_threshold(
                data,
                BehavioralCalibrationAgent(model_config, BENCHMARK),
                confidence_threshold,
                model_name,
                filename_id,
            )
            for confidence_threshold in CONFIDENCE_TARGETS
        ]

        # Wait for all tasks to complete
        threshold_results = await asyncio.gather(*tasks)

        overall_end = datetime.now()
        logger.info(
            "[%s] All threshold tasks completed in %.2f seconds",
            overall_end.strftime("%H:%M:%S.%f")[:-3],
            (overall_end - overall_start).total_seconds(),
        )

        # Convert results back to the expected format
        evaluation_results: Dict[float, List[ItemEval]] = {}
        for confidence_threshold, results in threshold_results:
            evaluation_results[confidence_threshold] = results

        file_path = (
            f"{OUTPUT_DIR}/{model_name}/{model_name}_predictions_{filename_id}.csv"
        )
        save_model_results(evaluation_results, file_path)
        logger.info("Results saved to %s", file_path)

        agg_results_by_model[model_name] = {}

        logger.info(f"\nModel: {model_name}")
        logger.info(
            "t\t\tAcc_beh\t\tCov_beh\t\tPay_beh\t\tConf_Correct\tConf_Incorrect"
        )
        for confidence_threshold in sorted(evaluation_results.keys()):
            rows = evaluation_results[confidence_threshold]
            acc_b, cov_b, pay_b = summarize_behavioral(rows)
            conf_metrics = confidence_by_correctness(rows)

            acc_b_str = (
                "NaN"
                if (isinstance(acc_b, float) and math.isnan(acc_b))
                else f"{acc_b:.3f}"
            )

            # Format confidence metrics
            conf_correct_str = (
                "NaN"
                if (
                    isinstance(conf_metrics["avg_conf_correct"], float)
                    and math.isnan(conf_metrics["avg_conf_correct"])
                )
                else f"{conf_metrics['avg_conf_correct']:.3f}"
            )
            conf_incorrect_str = (
                "NaN"
                if (
                    isinstance(conf_metrics["avg_conf_incorrect"], float)
                    and math.isnan(conf_metrics["avg_conf_incorrect"])
                )
                else f"{conf_metrics['avg_conf_incorrect']:.3f}"
            )

            logger.info(
                f"{confidence_threshold:.2f}\t{acc_b_str}\t\t{cov_b:.3f}\t\t{pay_b:.3f}\t\t"
                f"{conf_correct_str}\t\t{conf_incorrect_str}"
            )
            output.append(
                {
                    "model": model_name,
                    "target_confidence": confidence_threshold,
                    "acc_b": acc_b,
                    "cov_b": cov_b,
                    "pay_b": pay_b,
                    "avg_conf_correct": conf_metrics["avg_conf_correct"],
                    "avg_conf_incorrect": conf_metrics["avg_conf_incorrect"],
                }
            )

            agg_results_by_model[model_name][confidence_threshold] = {
                "acc_b": acc_b,
                "cov_b": cov_b,
                "pay_b": pay_b,
                "avg_conf_correct": conf_metrics["avg_conf_correct"],
                "avg_conf_incorrect": conf_metrics["avg_conf_incorrect"],
            }

    file_path = f"{OUTPUT_DIR}/aggregated/aggregated_results_{filename_id}.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(output).to_csv(file_path, index=False)
    logger.info("Results saved to %s", file_path)
    # Define schema for config serialization

    # Save config as JSON alongside results using schema
    config_json_path = f"{OUTPUT_DIR}/config/config_{filename_id}.json"
    os.makedirs(os.path.dirname(config_json_path), exist_ok=True)
    config = ConfigSchema(
        CONFIDENCE_TARGETS=CONFIDENCE_TARGETS,
        BENCHMARK=BENCHMARK,
        MODELS=MODELS,
        TEMPERATURE=TEMPERATURE,
    )
    with open(config_json_path, "w") as f:
        json.dump(config.model_dump(), f, indent=4, default=str)
    logger.info("Config saved to %s", config_json_path)

    plot_overlays(agg_results_by_model, filename_id)


if __name__ == "__main__":
    asyncio.run(main())
