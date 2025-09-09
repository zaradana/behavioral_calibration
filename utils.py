import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

from config import LOGS_DIR
from schema import ItemEval


def get_logger(name: str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f"{LOGS_DIR}/logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
        ],
    )
    return logging.getLogger(name)


def correctness_heuristic(answer: str, expect_substrings: List[str]) -> bool:
    """Simple heuristic to check if an answer contains expected substrings."""
    a = (answer or "").lower()
    return all(s in a for s in (s.lower() for s in expect_substrings))


def save_model_results(
    results: Dict[str, Dict[float, List[ItemEval]]], output_filename: str
) -> None:
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output = []
    for t, item_vals in results.items():
        for item_val in item_vals:
            output.append(
                {
                    "target_confidence": t,
                    "item_id": item_val.item_id,
                    "decision": item_val.decision,
                    "answer": item_val.answer,
                    "confidence": item_val.confidence,
                }
            )
    pd.DataFrame(output).to_csv(output_filename, index=False)
    logger.info("Results saved to %s", output_filename)
