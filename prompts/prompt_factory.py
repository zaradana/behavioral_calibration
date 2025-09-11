from prompts.behavioral_calibration_prompt import behavioral_calibration_prompt_template
from prompts.gpqa_prompt import get_gpqa_prompt
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

        if "swe" in benchmark_config.dataset_name.lower():
            return get_swe_prompt(
                processed_instance.prompt_data, behavioral_calibration_prompt
            )
        elif "gpqa" in benchmark_config.dataset_name.lower():
            return get_gpqa_prompt(
                processed_instance.prompt_data, behavioral_calibration_prompt
            )
        else:
            # For other datasets, default to proxy style prompt
            return get_proxy_prompt(
                processed_instance.prompt_data, behavioral_calibration_prompt
            )
