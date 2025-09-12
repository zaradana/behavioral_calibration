"""
CLI tool for model calibration analysis. This script analyzes model calibration by simulating
how accuracy and payoff change when the model abstains (says "idk") for predictions below different
confidence thresholds. It also compares simulated results with empirical results for specific
target confidence values.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from ..utils.calibration_utils import (
    CalibrationAnalyzer,
    calculate_confidence_statistics,
    load_and_validate_data,
)
from ..utils.core_utils import get_logger

# from ..reporting.calibration_plots import create_summary_dashboard

logger = get_logger(__name__)


def run_basic_analysis(csv_path: str, thresholds: Optional[List[float]] = None) -> None:
    """
    Run basic threshold analysis and logger.info results.

    Args:
        csv_path: Path to the CSV file
        thresholds: Optional list of thresholds to test
    """
    logger.info("🔍 Loading and validating data...")
    data = load_and_validate_data(csv_path)
    logger.info(f"✅ Loaded {len(data)} predictions")

    logger.info("🔍 Loading baseline data with target confidence = 0.0...")
    baseline_data = data[data["target_confidence"] == 0.0]

    logger.info("\n📊 Baseline data statistics:")
    total_predictions = len(baseline_data)
    answered_count = len(baseline_data[baseline_data["decision"] == "answer"])
    abstained_count = total_predictions - answered_count

    logger.info(f"   • Total predictions: {total_predictions}")
    logger.info(
        f"   • Answered: {answered_count} ({answered_count / total_predictions * 100:.1%})"
    )
    logger.info(
        f"   • Abstained: {abstained_count} ({abstained_count / total_predictions * 100:.1%})"
    )

    # Create analyzer
    analyzer = CalibrationAnalyzer(baseline_data)

    # Calculate confidence statistics
    conf_stats = calculate_confidence_statistics(baseline_data)
    logger.info("\n📈 Confidence statistics (answered questions):")
    logger.info(f"   • Mean confidence: {conf_stats['mean_confidence']}")
    logger.info(f"   • Median confidence: {conf_stats['median_confidence']}")
    logger.info(f"   • Std deviation: {conf_stats['confidence_std']}")
    logger.info(f"   • Range: {conf_stats['confidence_range']}")

    logger.info(f"\n🎯 Running threshold analysis for thresholds: {thresholds}")

    results = analyzer.run_threshold_sweep(thresholds)

    logger.info("")
    logger.info("📋 Threshold Analysis Results:")
    logger.info("=" * 100)

    # Create a nicely formatted table
    header = (
        f"{'Threshold':>8} {'Accuracy':>8} {'Coverage':>8} {'Abstention':>11} "
        f"{'Answered':>8} {'Total':>6} {'Payoff':>8} {'Conf_Correct':>12} {'Conf_Incorrect':>14}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for _, row in results.iterrows():
        logger.info(
            f"\t{row['threshold']:.2f}\t{row['accuracy']:.2f}\t{row['coverage']:.2f}\t\t"
            f"{row['abstention_rate']:.2f}\t{row['answered_questions']:.2f}\t{row['total_questions']:.2f}"
            f"\t{row['average_payoff']:.2f}\t{row['avg_conf_correct']:.2f}\t\t{row['avg_conf_incorrect']:.2f}"
        )

    logger.info("=" * 100)


def run_comparison_analysis(
    csv_path: str, thresholds: Optional[List[float]] = None
) -> None:
    """
    Run comparison analysis between empirical and simulated results.

    Args:
        csv_path: Path to the CSV file
    """

    logger.info("🔍 Loading data for comparison analysis...")
    data = load_and_validate_data(csv_path)
    analyzer = CalibrationAnalyzer(data)

    # Get empirical results
    empirical_results = analyzer.get_empirical_results()
    logger.info("📈 Empirical results:")
    if not empirical_results.empty:
        for _, row in empirical_results.iterrows():
            logger.info(
                f"   • Threshold {row['threshold']}: "
                f"Accuracy={row['accuracy']:.3f}, "
                f"Coverage={row['coverage']:.3f}, "
                f"Abstention={row['abstention_rate']:.3f}, "
                f"Answered={row['answered_questions']}, "
                f"Total={row['total_questions']}, "
                f"Payoff={row['average_payoff']:.3f}, "
                f"Conf Correct={row['avg_conf_correct']:.3f}, "
                f"Conf Incorrect={row['avg_conf_incorrect']:.3f}"
            )
    else:
        logger.info(
            "   ⚠️  No empirical data found for specified target confidence values"
        )
        return

    # Compare with simulation
    comparison_results = analyzer.compare_simulation_vs_empirical(thresholds)
    logger.info("🔬 Simulation vs Empirical comparison:")
    if not comparison_results.empty:
        for _, row in comparison_results.iterrows():
            logger.info(f"   • Threshold {row['threshold']}:")
            logger.info(
                f"     - Accuracy: Empirical={row['empirical_accuracy']:.3f}, "
                f"Simulated={row['simulated_accuracy']:.3f}, "
                f"Diff={row['accuracy_diff']:+.3f}"
            )
            logger.info(
                f"     - Payoff: Empirical={row['empirical_payoff']:.3f}, "
                f"Simulated={row['simulated_payoff']:.3f}, "
                f"Diff={row['payoff_diff']:+.3f}"
            )
            logger.info(
                f"     - Coverage: Empirical={row['empirical_coverage']:.3f}, "
                f"Simulated={row['simulated_coverage']:.3f}, "
                f"Diff={row['coverage_diff']:+.3f}"
            )
            logger.info(
                f"     - Abstention: Empirical={row['empirical_abstention_rate']:.3f}, "
                f"Simulated={row['simulated_abstention_rate']:.3f}, "
                f"Diff={row['abstention_diff']:+.3f}"
            )
            logger.info(
                f"     - Answered questions: Empirical={row['empirical_answered_questions']}, "
                f"Simulated={row['simulated_answered_questions']}, "
                f"Diff={row['answered_questions_diff']:+.3f}"
            )
            logger.info(
                f"     - Total questions: Empirical={row['empirical_total_questions']}, "
                f"Simulated={row['simulated_total_questions']}, "
                f"Diff={row['total_questions_diff']:+.3f}"
            )
            logger.info(
                f"     - Avg conf correct: Empirical={row['empirical_avg_conf_correct']:.3f}, "
                f"Simulated={row['simulated_avg_conf_correct']:.3f}, "
                f"Diff={row['avg_conf_correct_diff']:+.3f}"
            )
            logger.info(
                f"     - Avg conf incorrect: Empirical={row['empirical_avg_conf_incorrect']:.3f}, "
                f"Simulated={row['simulated_avg_conf_incorrect']:.3f}, "
                f"Diff={row['avg_conf_incorrect_diff']:+.3f}"
            )

    return comparison_results


# def run_full_report(csv_path: str, output_dir: str) -> None:
#     """
#     Generate a comprehensive analysis report with visualizations.

#     Args:
#         csv_path: Path to the CSV file
#         output_dir: Directory to save outputs
#     """
#     logger.info("🚀 Starting comprehensive calibration analysis...")

#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)

#     # Load data and create analyzer
#     data = load_and_validate_data(csv_path)
#     analyzer = CalibrationAnalyzer(data)

#     logger.info(f"📊 Data loaded: {len(data)} total predictions")
#     logger.info(
#         f"📊 Baseline data: {len(analyzer.baseline_data)} predictions (target_confidence = 0.0)"
#     )

#     # Run analyses
#     logger.info("\n🎯 Running threshold sweep (0% to 100%)...")
#     threshold_results = analyzer.run_threshold_sweep()
#     threshold_results.to_csv(output_path / "threshold_sweep_results.csv", index=False)

#     logger.info("📈 Extracting empirical results for target_confidence 0.5 and 0.8...")
#     empirical_results = analyzer.get_empirical_results([0.5, 0.8])
#     if not empirical_results.empty:
#         empirical_results.to_csv(output_path / "empirical_results.csv", index=False)

#     logger.info("🔬 Comparing simulation vs empirical results...")
#     comparison_results = analyzer.compare_simulation_vs_empirical([0.5, 0.8])
#     if not comparison_results.empty:
#         comparison_results.to_csv(
#             output_path / "simulation_vs_empirical.csv", index=False
#         )

#     # Generate visualizations
#     logger.info("📊 Creating visualizations...")
#     create_summary_dashboard(analyzer, str(output_path))

#     # logger.info summary
#     logger.info("\n" + "=" * 60)
#     logger.info("📋 ANALYSIS SUMMARY")
#     logger.info("=" * 60)

#     logger.info("\n🎯 Threshold Sweep Results:")
#     best_accuracy_idx = threshold_results["accuracy"].idxmax()
#     best_payoff_idx = threshold_results["average_payoff"].idxmax()
#     logger.info(
#         f"   • Best accuracy: {threshold_results.loc[best_accuracy_idx, 'accuracy']:.3f} at {threshold_results.loc[best_accuracy_idx, 'threshold']:.0f}% threshold"
#     )
#     logger.info(
#         f"   • Best payoff: {threshold_results.loc[best_payoff_idx, 'average_payoff']:.3f} at {threshold_results.loc[best_payoff_idx, 'threshold']:.0f}% threshold"
#     )

#     if not empirical_results.empty:
#         logger.info("\n📈 Empirical Results:")
#         for _, row in empirical_results.iterrows():
#             logger.info(
#                 f"   • Target {row['target_confidence']}: Accuracy={row['accuracy']:.3f}, Abstention={row['abstention_rate']:.3f}, Payoff={row['average_payoff']:.3f}"
#             )

#     if not comparison_results.empty:
#         logger.info("\n🔬 Key Findings:")
#         for _, row in comparison_results.iterrows():
#             logger.info(f"   • Target {row['target_confidence']}:")
#             logger.info(
#                 f"     - Simulation {'overestimates' if row['accuracy_diff'] < 0 else 'underestimates'} accuracy by {abs(row['accuracy_diff']):.3f}"
#             )
#             logger.info(f"     - Payoff difference: {row['payoff_diff']:+.3f}")

#     logger.info(
#         f"\n💾 All results and visualizations saved to: {output_path.absolute()}"
#     )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze model calibration from prediction CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with default thresholds
    python analyze_calibration.py data.csv --mode basic

    # Detailed threshold analysis
    python analyze_calibration.py data.csv --mode basic --thresholds 0 10 20 30 40 50

    # Compare simulation vs empirical
    python analyze_calibration.py data.csv --mode compare --targets 0.5 0.8

    # Full report with visualizations
    python analyze_calibration.py data.csv --mode report --output-dir results/
        """,
    )

    parser.add_argument("csv_path", help="Path to the CSV file with model predictions")
    parser.add_argument(
        "--mode",
        choices=["basic", "compare", "report"],
        default="basic",
        help="Analysis mode (default: basic)",
    )
    parser.add_argument(
        "--output-dir",
        default="calibration_analysis",
        help="Output directory for report mode (default: calibration_analysis)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Threshold values to test for basic mode (default: 0,10,20,...,100)",
    )

    args = parser.parse_args()

    # Validate CSV file exists
    if not Path(args.csv_path).exists():
        logger.info(f"❌ Error: File not found: {args.csv_path}")
        return 1

    if args.thresholds is not None:
        args.thresholds = list(range(1, 100, 5))

    try:
        if args.mode == "basic":
            run_basic_analysis(args.csv_path, args.thresholds)
        elif args.mode == "compare":
            run_comparison_analysis(args.csv_path, args.thresholds)
        # elif args.mode == "report":
        # run_full_report(args.csv_path, args.output_dir)

        logger.info("\n✅ Analysis completed successfully!")

    except Exception as e:
        logger.info(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
