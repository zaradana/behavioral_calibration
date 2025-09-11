import math
import os
from typing import Dict

import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from utils.core_utils import get_logger

logger = get_logger(__name__)


def plot_overlays(
    results_by_model: Dict[str, Dict[float, Dict[str, float]]],
    output_filename: str,
    benchmark_name: str,
) -> None:
    """
    Plots:
      (A) Accuracy vs t (two curves): behavioral vs confidence-only
      (B) Coverage vs t (two curves): behavioral vs confidence-only
      (C) Avg payoff vs t (behavioral only)
      (D) Average confidence by correctness (correct vs incorrect answers)
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_acc, ax_cov, ax_pay, ax_conf = axes.flatten()

    for model, by_t in results_by_model.items():
        ts = sorted(by_t.keys())

        # Collect series
        acc_beh, cov_beh, pay_beh = [], [], []
        conf_correct, conf_incorrect = [], []

        for t in ts:
            row = by_t[t]
            acc_beh.append(
                None
                if (isinstance(row["acc_b"], float) and math.isnan(row["acc_b"]))
                else row["acc_b"]
            )
            cov_beh.append(row["cov_b"])
            pay_beh.append(row["pay_b"])

            # Handle NaN values for confidence metrics
            conf_correct.append(
                None
                if (
                    isinstance(row["avg_conf_correct"], float)
                    and math.isnan(row["avg_conf_correct"])
                )
                else row["avg_conf_correct"]
            )
            conf_incorrect.append(
                None
                if (
                    isinstance(row["avg_conf_incorrect"], float)
                    and math.isnan(row["avg_conf_incorrect"])
                )
                else row["avg_conf_incorrect"]
            )

        # Drop NaNs for plotting accuracies
        ts_acc_beh = [t for t, a in zip(ts, acc_beh) if a is not None]
        acc_beh_pl = [a for a in acc_beh if a is not None]

        # (A) Accuracy
        ax_acc.plot(ts_acc_beh, acc_beh_pl, marker="o", label=f"{model} — behavioral")

        # (B) Coverage
        ax_cov.plot(ts, cov_beh, marker="s", label=f"{model} — behavioral")

        # (C) Payoff (behavioral)
        ax_pay.plot(ts, pay_beh, marker="^", label=model)

        # (D) Confidence by correctness
        # Filter out NaN values for plotting
        ts_conf_correct = [t for t, c in zip(ts, conf_correct) if c is not None]
        conf_correct_pl = [c for c in conf_correct if c is not None]

        ts_conf_incorrect = [t for t, c in zip(ts, conf_incorrect) if c is not None]
        conf_incorrect_pl = [c for c in conf_incorrect if c is not None]

        ax_conf.plot(
            ts_conf_correct,
            conf_correct_pl,
            marker="o",
            label=f"{model} — Correct answers",
        )
        ax_conf.plot(
            ts_conf_incorrect,
            conf_incorrect_pl,
            marker="s",
            linestyle="--",
            label=f"{model} — Incorrect answers",
        )

    ax_acc.set_title("Accuracy vs Target t ({benchmark_name})")
    ax_acc.set_xlabel("t")
    ax_acc.set_ylabel("Accuracy (answered only)")
    ax_acc.grid(True)
    ax_acc.legend(fontsize=8)

    ax_cov.set_title("Coverage vs Target t ({benchmark_name})")
    ax_cov.set_xlabel("t")
    ax_cov.set_ylabel("Coverage (fraction answered)")
    ax_cov.grid(True)
    ax_cov.legend(fontsize=8)

    ax_pay.set_title("Average Payoff vs Target t (Behavioral) ({benchmark_name})")
    ax_pay.set_xlabel("t")
    ax_pay.set_ylabel("Avg Payoff per Item")
    ax_pay.grid(True)
    ax_pay.legend(fontsize=8)

    ax_conf.set_title("Average Confidence by Correctness ({benchmark_name})")
    ax_conf.set_xlabel("Target Confidence Threshold (t)")
    ax_conf.set_ylabel("Average Confidence")
    ax_conf.grid(True)
    ax_conf.legend(fontsize=8)
    ax_conf.set_ylim(0, 1)  # Confidence is between 0 and 1

    plt.tight_layout()
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/plots/{output_filename}.png")
    plt.show()
    logger.info(f"Plots saved to {OUTPUT_DIR}/plots/{output_filename}.png")
