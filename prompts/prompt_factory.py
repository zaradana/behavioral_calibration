from prompts.behavioral_calibration_prompt import behavioral_calibration_prompt_template
from prompts.mc_prompt import get_mc_prompt
from prompts.proxy_prompt import get_proxy_prompt
from prompts.swe_prompt import prompt_style_2 as get_swe_prompt
from schema import BenchmarkConfig
from utils.instance_processor import ProcessedInstance


class PromptFactory:
    @staticmethod
    def get_prompt(
        benchmark_config: BenchmarkConfig,
        processed_instance: ProcessedInstance,
        target_threshold: float,
    ) -> str:
        behavioral_calibration_prompt = behavioral_calibration_prompt_template.format(
            target_threshold=target_threshold
        )
        prompt_registry = {
            "swe": get_swe_prompt,
            "gpqa": get_mc_prompt,
            "truthfulqa": get_mc_prompt,
            "proxy": get_proxy_prompt,
        }

        dataset_name = benchmark_config.dataset_name.lower()

        # Try exact match first
        if dataset_name in prompt_registry:
            return prompt_registry[dataset_name](
                processed_instance.prompt_data, behavioral_calibration_prompt
            )

        # Try partial matches for SWE-bench variants
        for pattern, prompt_func in prompt_registry.items():
            if pattern in dataset_name:
                return prompt_func(
                    processed_instance.prompt_data, behavioral_calibration_prompt
                )
        raise ValueError(
            f"No prompt found for benchmark {benchmark_config.dataset_name}"
        )
