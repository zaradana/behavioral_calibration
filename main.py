import pandas as pd
import math
import os
import asyncio
from typing import Dict, Any
from agent import get_agent
from evaluation import summarize_behavioral, inconsistency_rates, evaluate_model
from visualization import plot_overlays
from config import MODELS, OUTPUT_DIR, TARGETS
from datetime import datetime
from utils import get_logger, save_model_results
from schema import ModelConfig
from data_loader import DATA

logger = get_logger(__name__)


async def main():

    results_by_model: Dict[str, Dict[float, Dict[str, float]]] = {}
    output = []
    output_filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for model_config in MODELS:
        if not model_config.is_free:
            continue
        model_name = model_config.model_name
        logger.info("Evaluating %s ...", model_name)
        agent = get_agent(model_config)
        per_t = await evaluate_model(DATA, TARGETS, agent.run_evaluation)
        save_model_results(
            per_t, f"{OUTPUT_DIR}/{model_name}/{model_name}_{output_filename}.csv"
        )
        results_by_model[model_name] = {}

        logger.info(f"\nModel: {model_name}")
        logger.info("t\t\tAcc_beh\t\tCov_beh\t\tPay_beh\t\tIDK&High\tAns&Low")
        for t in sorted(per_t.keys()):
            rows = per_t[t]
            acc_b, cov_b, pay_b = summarize_behavioral(rows)
            incon = inconsistency_rates(rows, t)

            acc_b_str = (
                "NaN"
                if (isinstance(acc_b, float) and math.isnan(acc_b))
                else f"{acc_b:.3f}"
            )

            logger.info(
                f"{t:.2f}\t{acc_b_str}\t\t{cov_b:.3f}\t\t{pay_b:.3f}\t\t"
                f"{incon['idk_high_conf']:.3f}\t\t{incon['answer_low_conf']:.3f}"
            )
            output.append(
                {
                    "model": model_name,
                    "target_confidence": t,
                    "acc_b": acc_b,
                    "cov_b": cov_b,
                    "pay_b": pay_b,
                    "idk_high_conf": incon["idk_high_conf"],
                    "answer_low_conf": incon["answer_low_conf"],
                }
            )

            results_by_model[model_name][t] = {
                "acc_b": acc_b,
                "cov_b": cov_b,
                "pay_b": pay_b,
                "idk_high_conf": incon["idk_high_conf"],
                "answer_low_conf": incon["answer_low_conf"],
            }

    file_path = f"{OUTPUT_DIR}/{output_filename}.csv"
    pd.DataFrame(output).to_csv(file_path, index=False)
    logger.info("Results saved to %s", file_path)

    plot_overlays(results_by_model, output_filename)


if __name__ == "__main__":
    asyncio.run(main())
