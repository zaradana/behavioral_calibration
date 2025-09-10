from data_loader import get_data
from benchmarks import get_benchmark_config

proxy_benchmark_config = get_benchmark_config("proxy_data")
raw_proxy_dataset = get_data(benchmark_config=proxy_benchmark_config)

def correctness_heuristic(answer: str, item_id: str) -> bool:
    """Simple heuristic to check if an answer contains expected substrings."""
    a = (answer or "").lower()
    expect_substrings = next(item["expect_substrings"] for item in raw_proxy_dataset if item["id"] == item_id)
    return all(s in a for s in (s.lower() for s in expect_substrings))
