"""
CLI tool for model calibration assessment. This script evaluates model calibration by empirically
calculating how accuracy and payoff change when the model abstains (says "idk") for predictions
below different confidence thresholds.
"""

import argparse
import asyncio
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from ..core.agent import BehavioralCalibrationAgent
from ..core.benchmarks import Benchmarks, get_benchmark_config
from ..core.config import BENCHMARK, CONFIDENCE_TARGETS, MODELS, OUTPUT_DIR, TEMPERATURE
from ..core.data_loader import get_data
from ..core.evaluation import evaluate_model
from ..core.models import get_all_models, get_model_by_name
from ..core.schema import ConfigSchema, ItemEval
from ..reporting.visualization import plot_overlays
from ..utils.core_utils import get_logger, save_model_results
from ..utils.evaluation_utils import confidence_by_correctness, summarize_behavioral

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
    benchmark_config,
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
        benchmark_config=benchmark_config,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        filename_id=filename_id,
    )

    return confidence_threshold, evaluation_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Behavioral calibration evaluation for large language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Use config defaults
  python -m src.cli.assess_calibration

  # Evaluate specific models
  python -m src.cli.assess_calibration --models gpt-4o claude-3.5-sonnet

  # Use specific benchmark
  python -m src.cli.assess_calibration --benchmark gsm8k

  # Combine models and benchmark
  python -m src.cli.assess_calibration --models gpt-4o-mini --benchmark gpqa
        """,
    )

    # Model selection
    available_models = [model.model_name for model in get_all_models()]
    parser.add_argument(
        "--models",
        nargs="+",
        help=f"Models to evaluate. Available: {', '.join(available_models)}",
    )

    # Benchmark selection
    available_benchmarks = list(Benchmarks.keys())
    parser.add_argument(
        "--benchmark",
        help=f"Benchmark to use. Available: {', '.join(available_benchmarks)}",
    )

    return parser.parse_args()


async def main():
    # Parse command line arguments
    args = parse_arguments()

    # Determine which models to use (CLI args override config)
    if args.models:
        model_configs = [get_model_by_name(name) for name in args.models]
        # Filter out None values in case a model wasn't found
        model_configs = [config for config in model_configs if config is not None]
        if not model_configs:
            logger.error("No valid models found from CLI arguments")
            return
        logger.info(
            f"Using CLI-specified models: {[m.model_name for m in model_configs]}"
        )
    else:
        model_configs = MODELS
        logger.info(
            f"Using config default models: {[m.model_name for m in model_configs]}"
        )

    # Determine which benchmark to use (CLI args override config)
    if args.benchmark:
        benchmark_config = get_benchmark_config(args.benchmark)
        logger.info(f"Using CLI-specified benchmark: {args.benchmark}")
    else:
        benchmark_config = BENCHMARK
        logger.info(f"Using config default benchmark: {benchmark_config.dataset_name}")

    data = get_data(benchmark_config=benchmark_config)

    agg_results_by_model: Dict[str, Dict[float, Dict[str, float]]] = {}
    output = []
    filename_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for model_config in model_configs:
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
                BehavioralCalibrationAgent(model_config, benchmark_config),
                confidence_threshold,
                model_name,
                filename_id,
                benchmark_config,
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
        save_model_results(evaluation_results, file_path, benchmark_config.dataset_name)
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
                    "benchmark": benchmark_config.dataset_name,
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
    # Use the actual runtime configuration (from CLI args or config defaults)
    config_json_path = f"{OUTPUT_DIR}/config/config_{filename_id}.json"
    os.makedirs(os.path.dirname(config_json_path), exist_ok=True)
    config = ConfigSchema(
        CONFIDENCE_TARGETS=CONFIDENCE_TARGETS,
        BENCHMARK=benchmark_config,  # Use the actual benchmark used
        MODELS=model_configs,  # Use the actual models used
        TEMPERATURE=TEMPERATURE,
    )
    with open(config_json_path, "w") as f:
        json.dump(config.model_dump(), f, indent=4, default=str)
    logger.info("Config saved to %s", config_json_path)

    plot_overlays(agg_results_by_model, filename_id, benchmark_config.dataset_name)


if __name__ == "__main__":
    asyncio.run(main())
