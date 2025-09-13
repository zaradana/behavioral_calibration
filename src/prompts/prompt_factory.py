from ..core.schema import BenchmarkConfig
from ..utils.instance_processor import ProcessedInstance
from .behavioral_calibration_prompt import (
    behavioral_calibration_prompt_template,
    build_behavioral_calibration_prompt_with_abstinence_confidence_scoring,
    build_behavioral_calibration_prompt_with_abstinence_t_scoring,
)
from .gsm8k_prompt import get_gsm8k_prompt
from .mc_prompt import get_mc_prompt
from .proxy_prompt import get_proxy_prompt
from .svamp_prompt import get_svamp_prompt
from .swe_prompt import prompt_style_2 as get_swe_prompt


class PromptFactory:
    @staticmethod
    def get_prompt(
        benchmark_config: BenchmarkConfig,
        processed_instance: ProcessedInstance,
        target_threshold: float,
        scoring_method: str = "confidence",
    ) -> str:
        if target_threshold < 0.0:
            raise ValueError("Target threshold must be greater than 0.0")
        elif target_threshold > 0.0:
            if scoring_method == "confidence":
                behavioral_calibration_prompt = build_behavioral_calibration_prompt_with_abstinence_confidence_scoring(
                    target_threshold
                )
            elif scoring_method == "threshold":
                behavioral_calibration_prompt = (
                    build_behavioral_calibration_prompt_with_abstinence_t_scoring(
                        target_threshold
                    )
                )
            else:
                raise ValueError(f"Invalid scoring method: {scoring_method}")
            behavioral_calibration_prompt = (
                build_behavioral_calibration_prompt_with_abstinence_confidence_scoring(
                    target_threshold
                )
            )
        else:
            behavioral_calibration_prompt = behavioral_calibration_prompt_template

        prompt_registry = {
            "swe": get_swe_prompt,
            "gpqa": get_mc_prompt,
            "truthfulqa": get_mc_prompt,
            "proxy": get_proxy_prompt,
            "gsm8k": get_gsm8k_prompt,
            "svamp": get_svamp_prompt,
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
