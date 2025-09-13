"""
Utility functions for model calibration analysis.
"""

from typing import Dict, List

import pandas as pd

from ..core.schema import ItemEval
from ..utils.core_utils import get_logger
from .evaluation_utils import (
    confidence_by_correctness,
    payoff_value,
    summarize_behavioral,
)

logger = get_logger(__name__)


class CalibrationAnalyzer:
    """Core functionality for analyzing model calibration."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with prediction data.

        Args:
            data: DataFrame containing model predictions
        """
        self.data = data
        self.baseline_data = data[data["target_confidence"] == 0.0]
        logger.info(
            f"📊 Baseline data: {len(self.baseline_data)} predictions (target_confidence = 0.0)"
        )

        self.target_confidence_values = list(data["target_confidence"].unique())
        self.target_confidence_values.remove(0.0)
        logger.info(f"Target confidence values: {self.target_confidence_values}")
        self.fallback_thresholds = list(range(1, 100, 5))

    def simulate_threshold_effect(self, threshold: float) -> Dict[str, float]:
        """
        Simulate the effect of applying a confidence threshold.

        Args:
            threshold: Confidence threshold (0-100). Predictions below this become "idk"

        Returns:
            Dictionary with accuracy, abstention_rate, and payoff metrics
        """
        # Convert threshold from percentage to decimal
        threshold_decimal = threshold / 100.0

        # Create simulated decisions
        simulated_data = self.baseline_data.copy()

        # Replace answers with "idk" if confidence is below threshold
        low_confidence_mask = simulated_data["confidence"] <= threshold_decimal
        simulated_data.loc[low_confidence_mask, "decision"] = "idk"

        # Convert to ItemEval objects for helper functions
        items = [
            ItemEval(
                decision=row.decision,
                confidence=row.confidence,
                correct=row.correct,
                payoff_behavioral=payoff_value(
                    row.correct, row.decision, threshold_decimal
                ),
            )
            for _, row in simulated_data.iterrows()
        ]

        # Use helper functions to calculate metrics
        accuracy, coverage, avg_payoff = summarize_behavioral(items)
        conf_metrics = confidence_by_correctness(items)

        total_questions = len(items)
        answered_questions = sum(1 for item in items if item.decision == "answer")

        return {
            "threshold": threshold,
            "accuracy": accuracy,
            "coverage": coverage,
            "abstention_rate": 1 - coverage,
            "answered_questions": answered_questions,
            "total_questions": total_questions,
            "average_payoff": avg_payoff,
            "avg_conf_correct": conf_metrics["avg_conf_correct"],
            "avg_conf_incorrect": conf_metrics["avg_conf_incorrect"],
        }

    def run_threshold_sweep(self, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Run simulation across multiple threshold values.

        Args:
            thresholds: List of threshold values to test
        Returns:
            DataFrame with results for each threshold
        """
        if thresholds is None:
            thresholds = self.fallback_thresholds

        results = []
        for threshold in thresholds:
            result = self.simulate_threshold_effect(threshold)
            results.append(result)

        return pd.DataFrame(results)

    def get_empirical_results(self) -> pd.DataFrame:
        """
        Extract empirical results for specific target confidence values.

        Returns:
            DataFrame with empirical results

        """
        if len(self.target_confidence_values) == 0:
            raise ValueError("No target confidence values provided")

        empirical_results = []

        for target_conf in self.target_confidence_values:
            subset = self.data[self.data["target_confidence"] == target_conf]

            if subset.empty:
                continue

            # Convert subset to list of ItemEval objects
            eval_rows = [
                ItemEval(
                    decision=row["decision"],
                    correct=row["correct"],
                    confidence=row["confidence"],
                    payoff_behavioral=row["payoff_behavioral"],
                )
                for _, row in subset.iterrows()
            ]

            # Use evaluation utils functions
            accuracy, coverage, avg_payoff = summarize_behavioral(eval_rows)
            conf_metrics = confidence_by_correctness(eval_rows)

            total_questions = len(eval_rows)
            answered_questions = int(coverage * total_questions)

            empirical_results.append(
                {
                    "threshold": target_conf,
                    "accuracy": accuracy,
                    "coverage": coverage,
                    "abstention_rate": 1 - coverage,
                    "answered_questions": answered_questions,
                    "total_questions": total_questions,
                    "average_payoff": avg_payoff,
                    "avg_conf_correct": conf_metrics["avg_conf_correct"],
                    "avg_conf_incorrect": conf_metrics["avg_conf_incorrect"],
                }
            )

        return pd.DataFrame(empirical_results)

    def compare_simulation_vs_empirical(self, thresholds: List[float]) -> pd.DataFrame:
        """
        Compare simulation results with empirical results.

        Args:
            thresholds: Target confidence values to compare

        Returns:
            DataFrame comparing simulation vs empirical results
        """
        if len(self.target_confidence_values) == 0:
            raise ValueError("No target confidence values provided")

        empirical_df = self.get_empirical_results()
        simulation_results = self.run_threshold_sweep(thresholds)

        comparisons = []

        for _, emp_row in empirical_df.iterrows():
            target_conf = emp_row["threshold"]
            threshold_percent = target_conf * 100

            # Find closest simulation result
            sim_row = simulation_results.iloc[
                (simulation_results["threshold"] - threshold_percent)
                .abs()
                .argsort()[:1]
            ]
            sim_row = sim_row.iloc[0]

            comparison = {
                "threshold": target_conf,
                "threshold_percent": threshold_percent,
                "empirical_accuracy": emp_row["accuracy"],
                "simulated_accuracy": sim_row["accuracy"],
                "empirical_coverage": emp_row["coverage"],
                "simulated_coverage": sim_row["coverage"],
                "empirical_abstention_rate": emp_row["abstention_rate"],
                "simulated_abstention_rate": sim_row["abstention_rate"],
                "empirical_answered_questions": emp_row["answered_questions"],
                "simulated_answered_questions": sim_row["answered_questions"],
                "empirical_total_questions": emp_row["total_questions"],
                "simulated_total_questions": sim_row["total_questions"],
                "empirical_avg_conf_correct": emp_row["avg_conf_correct"],
                "simulated_avg_conf_correct": sim_row["avg_conf_correct"],
                "empirical_avg_conf_incorrect": emp_row["avg_conf_incorrect"],
                "simulated_avg_conf_incorrect": sim_row["avg_conf_incorrect"],
                "empirical_payoff": emp_row["average_payoff"],
                "simulated_payoff": sim_row["average_payoff"],
                "accuracy_diff": f"{emp_row['accuracy'] - sim_row['accuracy']:.3f} ({(emp_row['accuracy'] - sim_row['accuracy']) / emp_row['accuracy'] * 100:.2f}%)",
                "coverage_diff": f"{emp_row['coverage'] - sim_row['coverage']:.3f} ({(emp_row['coverage'] - sim_row['coverage']) / emp_row['coverage'] * 100:.2f}%)",
                "abstention_diff": f"{emp_row['abstention_rate'] - sim_row['abstention_rate']:.3f} ({(emp_row['abstention_rate'] - sim_row['abstention_rate']) / emp_row['abstention_rate'] * 100:.2f}%)",
                "answered_questions_diff": f"{emp_row['answered_questions'] - sim_row['answered_questions']:.3f} ({(emp_row['answered_questions'] - sim_row['answered_questions']) / emp_row['answered_questions'] * 100:.2f}%)",
                "total_questions_diff": f"{emp_row['total_questions'] - sim_row['total_questions']:.3f} ({(emp_row['total_questions'] - sim_row['total_questions']) / emp_row['total_questions'] * 100:.2f}%)",
                "avg_conf_correct_diff": f"{emp_row['avg_conf_correct'] - sim_row['avg_conf_correct']:.3f} ({(emp_row['avg_conf_correct'] - sim_row['avg_conf_correct']) / emp_row['avg_conf_correct'] * 100:.2f}%)",
                "avg_conf_incorrect_diff": f"{emp_row['avg_conf_incorrect'] - sim_row['avg_conf_incorrect']:.3f} ({(emp_row['avg_conf_incorrect'] - sim_row['avg_conf_incorrect']) / emp_row['avg_conf_incorrect'] * 100:.2f}%)",
                "abstention_rate_diff": f"{emp_row['abstention_rate'] - sim_row['abstention_rate']:.3f} ({(emp_row['abstention_rate'] - sim_row['abstention_rate']) / emp_row['abstention_rate'] * 100:.2f}%)",
                "payoff_diff": f"{emp_row['average_payoff'] - sim_row['average_payoff']:.3f} ({(emp_row['average_payoff'] - sim_row['average_payoff']) / emp_row['average_payoff'] * 100:.2f}%)",
            }
            comparisons.append(comparison)

        return pd.DataFrame(comparisons)


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the CSV data.

    Args:
        csv_path: Path to the CSV file containing model predictions

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If data is invalid or missing required columns
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = [
            "benchmark",
            "id",
            "target_confidence",
            "decision",
            "answer",
            "confidence",
            "correct",
            "payoff_behavioral",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")


def calculate_confidence_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate confidence-related statistics from the data.

    Args:
        data: DataFrame with prediction data

    Returns:
        Dictionary with confidence statistics
    """
    answered_data = data[data["decision"] == "answer"]

    if answered_data.empty:
        return {
            "mean_confidence": "0.00%",
            "median_confidence": "0.00%",
            "confidence_std": "0.00%",
            "confidence_range": "0.00%",
        }

    confidences = answered_data["confidence"]

    return {
        "mean_confidence": f"{100 * confidences.mean():.2f}%",
        "median_confidence": f"{100 * confidences.median():.2f}%",
        "confidence_std": f"{100 * confidences.std():.2f}%",
        "confidence_range": f"{100 * (confidences.max() - confidences.min()):.2f}%",
    }
