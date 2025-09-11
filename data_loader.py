from datasets import IterableDataset, load_dataset

from config import HF_HUB_TOKEN
from schema import BenchmarkConfig
from utils.core_utils import get_logger

logger = get_logger(__name__)

# SWE-style proxy problems (expand or swap for actual SWE-bench runner)
proxy_data = [
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


# Create a generator function for the IterableDataset
def proxy_data_generator():
    for item in proxy_data:
        yield item


# Create IterableDataset using from_generator
raw_proxy_dataset = IterableDataset.from_generator(proxy_data_generator)


def get_data(benchmark_config: BenchmarkConfig) -> IterableDataset:
    """Load data based on benchmark configuration.

    Args:
        benchmark_config: Configuration specifying dataset name, split, and whether to use proxy data

    Returns:
        List of data instances
    """
    try:
        # Login using e.g. `huggingface-cli login` to access this dataset
        # Only load dataset if not using proxy data
        if benchmark_config.dataset_name == "proxy_data":
            return raw_proxy_dataset

        raw_dataset = load_dataset(
            benchmark_config.dataset_path,
            benchmark_config.config_name,
            split=benchmark_config.dataset_split,
            token=HF_HUB_TOKEN,
        )
        return raw_dataset
    except Exception as e:
        print(f"Warning: Failed to load dataset {benchmark_config.dataset_name}: {e}")
        print("Falling back to proxy data")
        return raw_proxy_dataset
    # For SWE-bench and other HuggingFace datasets - convert to list
