import math
import os
from typing import Dict

import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from utils import get_logger

logger = get_logger(__name__)


def plot_overlays(
    results_by_model: Dict[str, Dict[float, Dict[str, float]]], output_filename: str
) -> None:
    """
    Plots:
      (A) Accuracy vs t (two curves): behavioral vs confidence-only
      (B) Coverage vs t (two curves): behavioral vs confidence-only
      (C) Avg payoff vs t (behavioral only)
      (D) Inconsistency vs t (stacked: idk_high_conf, answer_low_conf)
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_acc, ax_cov, ax_pay, ax_incon = axes.flatten()

    for model, by_t in results_by_model.items():
        ts = sorted(by_t.keys())

        # Collect series
        acc_beh, cov_beh, pay_beh = [], [], []
        incon_idk_high, incon_ans_low = [], []

        for t in ts:
            row = by_t[t]
            acc_beh.append(
                None
                if (isinstance(row["acc_b"], float) and math.isnan(row["acc_b"]))
                else row["acc_b"]
            )
            cov_beh.append(row["cov_b"])
            pay_beh.append(row["pay_b"])

            incon_idk_high.append(row["idk_high_conf"])
            incon_ans_low.append(row["answer_low_conf"])

        # Drop NaNs for plotting accuracies
        ts_acc_beh = [t for t, a in zip(ts, acc_beh) if a is not None]
        acc_beh_pl = [a for a in acc_beh if a is not None]

        # (A) Accuracy
        ax_acc.plot(ts_acc_beh, acc_beh_pl, marker="o", label=f"{model} — behavioral")

        # (B) Coverage
        ax_cov.plot(ts, cov_beh, marker="s", label=f"{model} — behavioral")

        # (C) Payoff (behavioral)
        ax_pay.plot(ts, pay_beh, marker="^", label=model)

        # (D) Inconsistency (stacked bars per model would be cluttered; lines instead)
        ax_incon.plot(ts, incon_idk_high, marker="d", label=f"{model} — IDK&conf≥t")
        ax_incon.plot(
            ts,
            incon_ans_low,
            marker="d",
            linestyle="--",
            label=f"{model} — ANSWER&conf<t",
        )

    ax_acc.set_title("Accuracy vs Target t")
    ax_acc.set_xlabel("t")
    ax_acc.set_ylabel("Accuracy (answered only)")
    ax_acc.grid(True)
    ax_acc.legend(fontsize=8)

    ax_cov.set_title("Coverage vs Target t")
    ax_cov.set_xlabel("t")
    ax_cov.set_ylabel("Coverage (fraction answered)")
    ax_cov.grid(True)
    ax_cov.legend(fontsize=8)

    ax_pay.set_title("Average Payoff vs Target t (Behavioral)")
    ax_pay.set_xlabel("t")
    ax_pay.set_ylabel("Avg Payoff per Item")
    ax_pay.grid(True)
    ax_pay.legend(fontsize=8)

    ax_incon.set_title("Inconsistency vs Target t")
    ax_incon.set_xlabel("t")
    ax_incon.set_ylabel("Fraction of items")
    ax_incon.grid(True)
    ax_incon.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/plots/{output_filename}.png")
    logger.info(f"Plots saved to {OUTPUT_DIR}/plots/{output_filename}.png")
