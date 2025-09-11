import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

from config import LOGS_DIR
from schema import ItemEval

# Global flag to ensure we only configure logging once
_logging_configured = False


def get_logger(name: str):
    global _logging_configured

    os.makedirs(LOGS_DIR, exist_ok=True)

    # Only configure logging once
    if not _logging_configured:
        # Clear any existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging
        logging.basicConfig(
            level=(
                logging.DEBUG
                if os.getenv("ENV", "dev").lower() == "dev"
                else logging.INFO
            ),
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    f"{LOGS_DIR}/logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
            ],
            force=True,  # Force reconfiguration
        )
        _logging_configured = True

    return logging.getLogger(name)


def save_model_results(
    results: Dict[str, Dict[float, List[ItemEval]]],
    output_filename: str,
    benchmark_name: str,
) -> None:
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output = []
    for t, item_vals in results.items():
        for item_val in item_vals:
            output.append(
                {
                    "benchmark": benchmark_name,
                    "id": item_val.id,
                    "target_confidence": t,
                    "decision": item_val.decision,
                    "answer": item_val.answer,
                    "confidence": item_val.confidence,
                    "correct": item_val.correct,
                    "payoff_behavioral": item_val.payoff_behavioral,
                }
            )
    pd.DataFrame(output).to_csv(output_filename, index=False)
