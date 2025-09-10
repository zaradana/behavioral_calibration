from typing import Dict
from prompts.behavioral_calibration_prompt import behavioral_calibration_prompt_template
from prompts.swe_prompt import prompt_style_2 as get_swe_prompt
from prompts.proxy_prompt import get_proxy_prompt
from schema import BenchmarkConfig

class PromptFactory:
    @staticmethod
    def get_prompt(benchmark_config: BenchmarkConfig, instance: Dict[str, str], target_threshold: float) -> str:
        behavioral_calibration_prompt = behavioral_calibration_prompt_template.format(target_threshold=target_threshold)
        
        # Use proxy data prompt if explicitly requested
        if benchmark_config.dataset_name == "proxy_data":
            return get_proxy_prompt(instance, behavioral_calibration_prompt)
        # Use SWE-bench prompt for SWE-bench datasets (covers swe_bench_lite, SWE-bench, etc.)
        elif "swe" in benchmark_config.dataset_name.lower():
            return get_swe_prompt(instance, behavioral_calibration_prompt)
        else:
            # For other datasets, default to proxy style prompt
            return get_proxy_prompt(instance, behavioral_calibration_prompt)
