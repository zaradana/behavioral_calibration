from typing import Dict, List

from datasets import load_dataset

from schema import BenchmarkConfig

# SWE-style proxy problems (expand or swap for actual SWE-bench runner)
raw_proxy_dataset = [
    {
        "id": "bug-001",
        "prompt": (
            "Task: Diagnose and propose a minimal fix.\n"
            "Code:\n"
            "def max_val(lst):\n"
            "    return max(lst)\n\n"
            "Failure: Should not throw on empty lists; returning None is acceptable.\n"
            "Explain the issue and suggest a minimal patch."
        ),
        "expect_substrings": ["empty", "none"],  # naive heuristic
    },
    {
        "id": "bug-002",
        "prompt": (
            "Task: Diagnose and propose a minimal fix.\n"
            "Code:\n"
            "def add_items(x, items=[]):\n"
            "    items.append(x)\n"
            "    return items\n\n"
            "Failure: Repeated calls share state.\n"
            "Explain the issue and suggest a minimal patch."
        ),
        "expect_substrings": ["mutable default", "none", "initialize"],
    },
    # Add more items or wire up a repo+tests harness
]


def get_data(benchmark_config: BenchmarkConfig) -> List[Dict[str, str]]:
    """Load data based on benchmark configuration.

    Args:
        benchmark_config: Configuration specifying dataset name, split, and whether to use proxy data

    Returns:
        List of data instances
    """
    try:
        # Login using e.g. `huggingface-cli login` to access this dataset
        # Only load dataset if not using proxy data
        raw_dataset = load_dataset(
            benchmark_config.dataset_path, split=benchmark_config.dataset_split, config_name=benchmark_config.config_name
        )
    except Exception as e:
        print(f"Warning: Failed to load dataset {benchmark_config.dataset_path}: {e}")
        print("Falling back to proxy data")
        raw_dataset = None

    # Use proxy data if explicitly requested or if dataset loading failed
    if benchmark_config.dataset_name == "proxy_data" or raw_dataset is None:
        return raw_proxy_dataset

    # For SWE-bench and other HuggingFace datasets - convert to list
    return list(raw_dataset)
