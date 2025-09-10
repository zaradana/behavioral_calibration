"""
Predefined benchmark configurations for easy selection.
Usage: Import and modify config.py to use these predefined configurations.
"""

from schema import BenchmarkConfig

# Predefined benchmark configurations
Benchmarks = {
    "swe_bench_lite": BenchmarkConfig(
        dataset_name="swe_bench_lite",
        dataset_path="SWE-bench/SWE-bench_Lite",
        dataset_split="test",
        instance_id_key="instance_id",
        config_name="",
    ),
    "proxy_data": BenchmarkConfig(
        dataset_name="proxy_data",
        dataset_path="",
        dataset_split="test",
        instance_id_key="id",
        config_name="",
    ),
    "GPQA": BenchmarkConfig(
        dataset_name="GPQA",
        dataset_path="Idavidrein/gpqa",
        dataset_split="train",
        instance_id_key="id",
        config_name="gpqa_diamond",
    ),
}

def get_benchmark_config(name: str) -> BenchmarkConfig:
    """Get a predefined benchmark configuration by name."""
    if name not in Benchmarks:
        available = list(Benchmarks.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return Benchmarks[name]
