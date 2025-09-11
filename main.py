import asyncio
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from agent import BehavioralCalibrationAgent, get_agent
from config import BENCHMARK, CONFIDENCE_TARGETS, MODELS, OUTPUT_DIR, TEMPERATURE
from data_loader import get_data
from evaluation import evaluate_model
from schema import ConfigSchema, ItemEval
from utils.core_utils import get_logger, save_model_results
from utils.evaluation_utils import inconsistency_rates, summarize_behavioral
from visualization import plot_overlays

logger = get_logger(__name__)


async def get_predictions(
    data: List[Dict[str, Any]],
    agent: BehavioralCalibrationAgent,
    confidence_threshold: float,
) -> List[ItemEval]:
    predictions: List[ItemEval] = []
    for item in data:
        obj = await agent.run(item, confidence_threshold, BENCHMARK)
        predictions.append(
            ItemEval(
                decision=obj.decision,
                answer=obj.answer,
                confidence=obj.confidence,
                evaluation_metadata=obj.evaluation_metadata,
            )
        )
    return predictions


async def main():
    data = get_data(benchmark_config=BENCHMARK)

    agg_results_by_model: Dict[str, Dict[float, Dict[str, float]]] = {}
    output = []
    filename_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for model_config in MODELS:
        model_name = model_config.model_name
        agent = get_agent(model_config)
        evaluation_results: Dict[float, List[ItemEval]] = {}
        for confidence_threshold in CONFIDENCE_TARGETS:
            logger.info(
                "Getting predictions for %s with confidence threshold %s",
                model_name,
                confidence_threshold,
            )
            predictions: List[ItemEval] = await get_predictions(
                data, agent, confidence_threshold
            )

            logger.info("Evaluating %s ...", model_name)

            # Run evaluation with optional SWE-bench integration
            evaluation_results[confidence_threshold] = await evaluate_model(
                predictions=predictions,
                benchmark_config=BENCHMARK,
                confidence_threshold=confidence_threshold,
                model_name=model_name,
                filename_id=filename_id,
            )

        file_path = (
            f"{OUTPUT_DIR}/{model_name}/{model_name}_predictions_{filename_id}.csv"
        )
        save_model_results(evaluation_results, file_path)
        logger.info("Results saved to %s", file_path)

        agg_results_by_model[model_name] = {}

        logger.info(f"\nModel: {model_name}")
        logger.info("t\t\tAcc_beh\t\tCov_beh\t\tPay_beh\t\tIDK&High\tAns&Low")
        for confidence_threshold in sorted(evaluation_results.keys()):
            rows = evaluation_results[confidence_threshold]
            acc_b, cov_b, pay_b = summarize_behavioral(rows)
            incon = inconsistency_rates(rows, confidence_threshold)

            acc_b_str = (
                "NaN"
                if (isinstance(acc_b, float) and math.isnan(acc_b))
                else f"{acc_b:.3f}"
            )

            logger.info(
                f"{confidence_threshold:.2f}\t{acc_b_str}\t\t{cov_b:.3f}\t\t{pay_b:.3f}\t\t"
                f"{incon['idk_high_conf']:.3f}\t\t{incon['answer_low_conf']:.3f}"
            )
            output.append(
                {
                    "model": model_name,
                    "target_confidence": confidence_threshold,
                    "acc_b": acc_b,
                    "cov_b": cov_b,
                    "pay_b": pay_b,
                    "idk_high_conf": incon["idk_high_conf"],
                    "answer_low_conf": incon["answer_low_conf"],
                }
            )

            agg_results_by_model[model_name][confidence_threshold] = {
                "acc_b": acc_b,
                "cov_b": cov_b,
                "pay_b": pay_b,
                "idk_high_conf": incon["idk_high_conf"],
                "answer_low_conf": incon["answer_low_conf"],
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
