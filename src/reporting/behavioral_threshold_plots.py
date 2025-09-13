import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..core.config import OUTPUT_DIR
from ..core.schema import ItemEval
from ..utils.core_utils import get_logger

logger = get_logger(__name__)


def calculate_calibration_curve(
    evaluation_results: List[ItemEval], n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve data for confidence vs accuracy plot.

    Args:
        evaluation_results: List of ItemEval objects with predictions
        n_bins: Number of confidence bins to use

    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_counts)
    """
    # Filter to only answered questions
    answered = [r for r in evaluation_results if r.decision == "answer"]

    if not answered:
        # Return empty arrays if no answered questions
        return np.array([]), np.array([]), np.array([])

    confidences = np.array([r.confidence for r in answered])
    correctness = np.array([r.correct for r in answered])

    # Create confidence bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        # Find predictions in this bin
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        # Handle the last bin to include confidence = 1.0
        if i == n_bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        bin_data = correctness[in_bin]

        if len(bin_data) > 0:
            accuracy = bin_data.mean()
            count = len(bin_data)
        else:
            accuracy = 0
            count = 0

        bin_accuracies.append(accuracy)
        bin_counts.append(count)

    return bin_centers, np.array(bin_accuracies), np.array(bin_counts)


def plot_overlays(
    results_by_model: Dict[str, Dict[float, Dict[str, float]]],
    evaluation_results_by_model: Dict[str, Dict[float, List[ItemEval]]],
    output_filename: str,
    benchmark_name: str,
) -> None:
    """
    Plots:
      (A) Accuracy vs t (two curves): behavioral vs confidence-only
      (B) Coverage vs t (two curves): behavioral vs confidence-only
      (C) Avg payoff vs t (behavioral only)
      (D) Calibration plot (confidence vs accuracy)
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_acc, ax_cov, ax_pay, ax_conf = axes.flatten()

    for model, by_t in results_by_model.items():
        ts = sorted(by_t.keys())

        # Collect series
        acc_beh, cov_beh, pay_beh = [], [], []

        for t in ts:
            row = by_t[t]
            acc_beh.append(
                None
                if (isinstance(row["acc_b"], float) and math.isnan(row["acc_b"]))
                else row["acc_b"]
            )
            cov_beh.append(row["cov_b"])
            pay_beh.append(row["pay_b"])

        # Drop NaNs for plotting accuracies
        ts_acc_beh = [t for t, a in zip(ts, acc_beh) if a is not None]
        acc_beh_pl = [a for a in acc_beh if a is not None]

        # (A) Accuracy
        ax_acc.plot(ts_acc_beh, acc_beh_pl, marker="o", label=f"{model} — behavioral")

        # (B) Coverage
        ax_cov.plot(ts, cov_beh, marker="s", label=f"{model} — behavioral")

        # (C) Payoff (behavioral)
        ax_pay.plot(ts, pay_beh, marker="^", label=model)

    # (D) Calibration plot - use ONLY baseline data (t=0.0) for calibration analysis
    # This shows the model's natural confidence calibration without threshold filtering
    for model in evaluation_results_by_model:
        if 0.0 in evaluation_results_by_model[model]:
            baseline_results = evaluation_results_by_model[model][0.0]
            bin_centers, bin_accuracies, bin_counts = calculate_calibration_curve(
                baseline_results
            )

            # Only plot if we have data
            if len(bin_centers) > 0:
                # Filter out bins with zero counts to avoid plotting empty bins
                valid_bins = bin_counts > 0
                if np.any(valid_bins):
                    ax_conf.plot(
                        bin_centers[valid_bins],
                        bin_accuracies[valid_bins],
                        marker="o",
                        label=f"{model}",
                        linewidth=2,
                        markersize=6,
                    )

    ax_acc.set_title(f"Accuracy vs Target t ({benchmark_name})")
    ax_acc.set_xlabel("t")
    ax_acc.set_ylabel("Accuracy (answered only)")
    ax_acc.grid(True)
    ax_acc.legend(fontsize=8)

    ax_cov.set_title(f"Coverage vs Target t ({benchmark_name})")
    ax_cov.set_xlabel("t")
    ax_cov.set_ylabel("Coverage (fraction answered)")
    ax_cov.grid(True)
    ax_cov.legend(fontsize=8)

    ax_pay.set_title(f"Average Payoff vs Target t (Behavioral) ({benchmark_name})")
    ax_pay.set_xlabel("t")
    ax_pay.set_ylabel("Avg Payoff per Item")
    ax_pay.grid(True)
    ax_pay.legend(fontsize=8)

    # Add perfect calibration line (diagonal)
    ax_conf.plot(
        [0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration", linewidth=1
    )

    ax_conf.set_title(f"Calibration Plot ({benchmark_name})")
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Accuracy")
    ax_conf.grid(True)
    ax_conf.legend(fontsize=8)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/plots/{output_filename}.png")
    plt.show()
    logger.info(f"Plots saved to {OUTPUT_DIR}/plots/{output_filename}.png")
