"""
Visualization functions for model calibration analysis.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for consistent plotting
plt.style.use("default")
sns.set_palette("husl")


def plot_threshold_sweep(
    results_df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """
    Plot the results of the threshold sweep.

    Args:
        results_df: DataFrame with threshold sweep results
        save_path: Optional path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy vs Threshold
    ax1.plot(
        results_df["threshold"],
        results_df["accuracy"],
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.set_xlabel("Confidence Threshold (%)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Confidence Threshold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Abstention Rate vs Threshold
    ax2.plot(
        results_df["threshold"],
        results_df["abstention_rate"],
        "r-",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax2.set_xlabel("Confidence Threshold (%)")
    ax2.set_ylabel("Abstention Rate")
    ax2.set_title("Abstention Rate vs Confidence Threshold")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Average Payoff vs Threshold
    ax3.plot(
        results_df["threshold"],
        results_df["average_payoff"],
        "g-",
        linewidth=2,
        marker="^",
        markersize=4,
    )
    ax3.set_xlabel("Confidence Threshold (%)")
    ax3.set_ylabel("Average Payoff")
    ax3.set_title("Average Payoff vs Confidence Threshold")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Answered Questions vs Threshold
    ax4.plot(
        results_df["threshold"],
        results_df["answered_questions"],
        "m-",
        linewidth=2,
        marker="d",
        markersize=4,
    )
    ax4.set_xlabel("Confidence Threshold (%)")
    ax4.set_ylabel("Number of Answered Questions")
    ax4.set_title("Answered Questions vs Confidence Threshold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Threshold sweep plot saved to: {save_path}")

    plt.show()


def plot_comparison(
    comparison_df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between empirical and simulated results.

    Args:
        comparison_df: DataFrame with comparison results
        save_path: Optional path to save the plot
    """
    if comparison_df.empty:
        print("⚠️  No comparison data to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    x = comparison_df["threshold"]
    width = 0.10

    # Accuracy comparison
    ax1.bar(
        x - width / 2,
        comparison_df["empirical_accuracy"],
        width,
        label="Empirical",
        alpha=0.8,
        color="skyblue",
    )
    ax1.bar(
        x + width / 2,
        comparison_df["simulated_accuracy"],
        width,
        label="Simulated",
        alpha=0.8,
        color="lightcoral",
    )
    ax1.set_xlabel("Target Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy: Empirical vs Simulated")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Abstention rate comparison
    ax2.bar(
        x - width / 2,
        comparison_df["empirical_abstention_rate"],
        width,
        label="Empirical",
        alpha=0.8,
        color="lightgreen",
    )
    ax2.bar(
        x + width / 2,
        comparison_df["simulated_abstention_rate"],
        width,
        label="Simulated",
        alpha=0.8,
        color="orange",
    )
    ax2.set_xlabel("Target Confidence")
    ax2.set_ylabel("Abstention Rate")
    ax2.set_title("Abstention Rate: Empirical vs Simulated")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Payoff comparison
    ax3.bar(
        x - width / 2,
        comparison_df["empirical_payoff"],
        width,
        label="Empirical",
        alpha=0.8,
        color="plum",
    )
    ax3.bar(
        x + width / 2,
        comparison_df["simulated_payoff"],
        width,
        label="Simulated",
        alpha=0.8,
        color="gold",
    )
    ax3.set_xlabel("Target Confidence")
    ax3.set_ylabel("Average Payoff")
    ax3.set_title("Average Payoff: Empirical vs Simulated")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Coverage comparison
    ax4.bar(
        x - width / 2,
        comparison_df["empirical_coverage"],
        width,
        label="Empirical",
        alpha=0.8,
        color="lightblue",
    )
    ax4.bar(
        x + width / 2,
        comparison_df["simulated_coverage"],
        width,
        label="Simulated",
        alpha=0.8,
        color="lightcoral",
    )
    ax4.set_xlabel("Target Confidence")
    ax4.set_ylabel("Coverage")
    ax4.set_title("Coverage: Empirical vs Simulated")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Comparison plot saved to: {save_path}")

    plt.show()


def plot_confidence_distribution(
    data: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """
    Plot confidence distribution for different decision types and correctness.

    Args:
        data: DataFrame with prediction data
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Filter data for baseline (target_confidence = 0.0)
    baseline_data = data[data["target_confidence"] == 0.0]
    answered_data = baseline_data[baseline_data["decision"] == "answer"]

    if answered_data.empty:
        print("⚠️  No answered data to plot confidence distribution")
        return

    # 1. Overall confidence distribution
    ax1.hist(
        answered_data["confidence"],
        bins=30,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Overall Confidence Distribution (Answered Questions)")
    ax1.grid(True, alpha=0.3)

    # 2. Confidence by correctness
    correct_conf = answered_data[answered_data["correct"]]["confidence"]
    incorrect_conf = answered_data[~answered_data["correct"]]["confidence"]

    # Use the same bin edges for both histograms to ensure consistent widths
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    ax2.hist(
        correct_conf,
        bins=bins,
        alpha=0.7,
        label="Correct",
        color="lightgreen",
        edgecolor="black",
    )
    ax2.hist(
        incorrect_conf,
        bins=bins,
        alpha=0.7,
        label="Incorrect",
        color="lightcoral",
        edgecolor="black",
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Confidence Distribution by Correctness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy vs confidence bins
    conf_bins = np.linspace(0, 1, 11)  # 10 bins
    bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []

    for i in range(len(conf_bins) - 1):
        bin_data = answered_data[
            (answered_data["confidence"] >= conf_bins[i])
            & (answered_data["confidence"] < conf_bins[i + 1])
        ]
        if not bin_data.empty:
            accuracy = bin_data["correct"].mean()
            count = len(bin_data)
        else:
            accuracy = 0
            count = 0
        bin_accuracies.append(accuracy)
        bin_counts.append(count)

    # Plot accuracy vs confidence
    ax3.bar(
        bin_centers,
        bin_accuracies,
        width=0.08,
        alpha=0.7,
        color="orange",
        edgecolor="black",
    )
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax3.set_xlabel("Confidence Bin")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Calibration Plot (Accuracy vs Confidence)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Confidence distribution plot saved to: {save_path}")

    plt.show()


def plot_coverage_analysis(
    results_df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """
    Create detailed coverage analysis plots.

    Args:
        results_df: DataFrame with threshold sweep results
        save_path: Optional path to save the plot
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Coverage vs Accuracy trade-off
    ax1.scatter(
        results_df["accuracy"],
        results_df["coverage"],
        c=results_df["threshold"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage vs Accuracy Trade-off")
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label("Threshold (%)")

    # 2. Coverage vs Payoff trade-off
    ax2.scatter(
        results_df["average_payoff"],
        results_df["coverage"],
        c=results_df["threshold"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax2.set_xlabel("Average Payoff")
    ax2.set_ylabel("Coverage")
    ax2.set_title("Coverage vs Payoff Trade-off")
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label("Threshold (%)")

    # 3. Efficiency frontier (Pareto optimal points)
    # Find points that are not dominated by any other point
    pareto_points = []
    for i, row in results_df.iterrows():
        is_pareto = True
        for j, other_row in results_df.iterrows():
            if (
                other_row["accuracy"] >= row["accuracy"]
                and other_row["coverage"] >= row["coverage"]
                and (
                    other_row["accuracy"] > row["accuracy"]
                    or other_row["coverage"] > row["coverage"]
                )
            ):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(i)

    # Color-code all points by threshold
    ax3.scatter(
        results_df["accuracy"],
        results_df["coverage"],
        color="lightblue",
        s=30,
        alpha=0.6,
        label="All points",
    )

    # Add threshold labels to all points with smart positioning
    # Group nearby points and offset their labels
    positions = []
    for idx, row in results_df.iterrows():
        x, y = row["accuracy"], row["coverage"]

        # Check for nearby points and calculate offset
        offset_x, offset_y = 5, 5
        for prev_x, prev_y, prev_offset_x, prev_offset_y in positions:
            # If points are close (within 0.02 units), offset the label
            if abs(x - prev_x) < 0.02 and abs(y - prev_y) < 0.02:
                offset_x = prev_offset_x + 15  # Move right
                if offset_x > 35:  # If too far right, move to next row
                    offset_x = 5
                    offset_y = prev_offset_y + 12  # Move down

        positions.append((x, y, offset_x, offset_y))

        ax3.annotate(
            f"{row['threshold']:.0f}",
            (x, y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

    if pareto_points:
        pareto_df = results_df.iloc[pareto_points]
        ax3.scatter(
            pareto_df["accuracy"],
            pareto_df["coverage"],
            color="red",
            s=80,
            label="Pareto optimal",
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
        )
    ax3.set_xlabel("Accuracy")
    ax3.set_ylabel("Coverage")
    ax3.set_title("Efficiency Frontier (Pareto Optimal Points)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Coverage analysis plot saved to: {save_path}")

    plt.show()


def plot_payoff_analysis(
    results_df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """
    Create detailed payoff analysis plots.

    Args:
        results_df: DataFrame with threshold sweep results
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Payoff vs Accuracy trade-off
    ax1.scatter(
        results_df["accuracy"],
        results_df["average_payoff"],
        c=results_df["threshold"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Average Payoff")
    ax1.set_title("Payoff vs Accuracy Trade-off")
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label("Threshold (%)")

    # 2. Payoff vs Coverage trade-off
    coverage = 1 - results_df["abstention_rate"]
    ax2.scatter(
        coverage,
        results_df["average_payoff"],
        c=results_df["threshold"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax2.set_xlabel("Coverage (1 - Abstention Rate)")
    ax2.set_ylabel("Average Payoff")
    ax2.set_title("Payoff vs Coverage Trade-off")
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label("Threshold (%)")

    # 3. Efficiency frontier (Pareto optimal points)
    # Find points that are not dominated by any other point
    pareto_points = []
    for i, row in results_df.iterrows():
        is_pareto = True
        for j, other_row in results_df.iterrows():
            if (
                other_row["accuracy"] >= row["accuracy"]
                and other_row["average_payoff"] >= row["average_payoff"]
                and (
                    other_row["accuracy"] > row["accuracy"]
                    or other_row["average_payoff"] > row["average_payoff"]
                )
            ):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(i)

    # Color-code all points by threshold
    ax3.scatter(
        results_df["accuracy"],
        results_df["average_payoff"],
        color="lightblue",
        s=30,
        alpha=0.6,
        label="All points",
    )

    # Add threshold labels to all points with smart positioning
    # Group nearby points and offset their labels
    positions = []
    for idx, row in results_df.iterrows():
        x, y = row["accuracy"], row["average_payoff"]

        # Check for nearby points and calculate offset
        offset_x, offset_y = 5, 5
        for prev_x, prev_y, prev_offset_x, prev_offset_y in positions:
            # If points are close (within 0.02 units for x, 0.1 for y), offset the label
            if abs(x - prev_x) < 0.02 and abs(y - prev_y) < 0.1:
                offset_x = prev_offset_x + 15  # Move right
                if offset_x > 35:  # If too far right, move to next row
                    offset_x = 5
                    offset_y = prev_offset_y + 12  # Move down

        positions.append((x, y, offset_x, offset_y))

        ax3.annotate(
            f"{row['threshold']:.0f}",
            (x, y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

    if pareto_points:
        pareto_df = results_df.iloc[pareto_points]
        ax3.scatter(
            pareto_df["accuracy"],
            pareto_df["average_payoff"],
            color="red",
            s=80,
            label="Pareto optimal",
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
        )
    ax3.set_xlabel("Accuracy")
    ax3.set_ylabel("Average Payoff")
    ax3.set_title("Efficiency Frontier (Pareto Optimal Points)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Payoff analysis plot saved to: {save_path}")

    plt.show()


def create_summary_dashboard(
    analyzer, thresholds: List[float], output_dir: str, file_stem: str
) -> None:
    """
    Create a comprehensive dashboard with all key visualizations.

    Args:
        analyzer: CalibrationAnalyzer instance
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("📊 Creating comprehensive visualization dashboard...")

    # Run analysis
    threshold_results = analyzer.run_threshold_sweep()
    comparison_results = analyzer.compare_simulation_vs_empirical(thresholds)

    # Create plots
    plot_threshold_sweep(
        threshold_results, str(output_path / f"{file_stem}_threshold_sweep.png")
    )
    plot_comparison(
        comparison_results, str(output_path / f"{file_stem}_empirical_vs_simulated.png")
    )
    plot_confidence_distribution(
        analyzer.data, str(output_path / f"{file_stem}_confidence_distribution.png")
    )
    plot_coverage_analysis(
        threshold_results, str(output_path / f"{file_stem}_coverage_analysis.png")
    )
    plot_payoff_analysis(
        threshold_results, str(output_path / f"{file_stem}_payoff_analysis.png")
    )

    print(f"✅ All visualizations saved to: {output_path.absolute()}")
