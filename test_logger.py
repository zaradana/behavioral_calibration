#!/usr/bin/env python3
"""Test script to verify the logger is working correctly."""

import sys

sys.path.append("/Users/zara/Document_Mac16/behavioral_calibration")

from utils.core_utils import get_logger


def test_logger():
    """Test that the logger writes to both console and file."""
    print("Testing logger configuration...")

    # Get a logger
    logger = get_logger("test_logger")

    # Test different log levels
    logger.info("This is an info message - should appear in both console and file")
    logger.warning("This is a warning message - should appear in both console and file")
    logger.error("This is an error message - should appear in both console and file")

    print("\nLogger test completed!")
    print("Check the logs/ directory for the log file.")

    # List log files to show which one was created
    from pathlib import Path

    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("logs_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"Latest log file: {latest_log}")

            # Show content of the log file
            with open(latest_log, "r") as f:
                content = f.read()
                if content:
                    print(f"\nLog file content:\n{content}")
                else:
                    print("Log file is empty!")
        else:
            print("No log files found in logs/ directory")
    else:
        print("logs/ directory does not exist")


if __name__ == "__main__":
    test_logger()
