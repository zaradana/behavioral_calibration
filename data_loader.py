from datasets import load_dataset

from config import BENCHMARK_DATASET

# Login using e.g. `huggingface-cli login` to access this dataset
DATA_REAL = load_dataset(BENCHMARK_DATASET, split="test")

# SWE-style proxy problems (expand or swap for actual SWE-bench runner)
DATA_PROXY = [
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

DATA = DATA_REAL if not DEV_MODE else DATA_PROXY
