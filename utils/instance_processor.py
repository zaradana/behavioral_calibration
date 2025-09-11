import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from schema import BenchmarkConfig, ProcessedInstance


class InstanceProcessor(ABC):
    """Base class for processing benchmark instances."""

    @abstractmethod
    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process a raw instance into prompt data and evaluation metadata."""
        pass


class GPQAInstanceProcessor(InstanceProcessor):
    """Processor for GPQA instances that handles answer shuffling."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process GPQA instance by shuffling answers and tracking correct index."""
        # Shuffle the answer choices
        example_choices = [
            instance["Incorrect Answer 1"],
            instance["Incorrect Answer 2"],
            instance["Incorrect Answer 3"],
            instance["Correct Answer"],
        ]

        # Set seed for reproducible shuffling per instance if provided
        if self.seed is not None:
            # Use instance-specific seed for reproducibility
            instance_seed = hash(instance.get("Question", "")) % (2**32)
            random.seed(self.seed + instance_seed)

        random.shuffle(example_choices)

        # Find the correct answer index after shuffling
        correct_index = example_choices.index(instance["Correct Answer"])

        # Restore global seed state if we were using instance-specific seeding
        if self.seed is not None:
            random.seed(self.seed)

        prompt_data = {
            "question": instance["Question"],
            "choice1": example_choices[0],
            "choice2": example_choices[1],
            "choice3": example_choices[2],
            "choice4": example_choices[3],
        }

        evaluation_metadata = {"correct_index": correct_index}

        return ProcessedInstance(
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )


class SWEInstanceProcessor(InstanceProcessor):
    """Processor for SWE-bench instances."""

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process SWE instance - mostly pass-through since no shuffling needed."""
        prompt_data = {
            "problem_statement": instance.get("problem_statement", ""),
            "readmes": instance.get("readmes", {}),
            "file_contents": instance.get("file_contents", {}),
        }

        evaluation_metadata = {
            "instance_id": instance.get("instance_id", ""),
        }

        return ProcessedInstance(
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )


class ProxyInstanceProcessor(InstanceProcessor):
    """Processor for proxy data instances."""

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process proxy instance - mostly pass-through."""
        prompt_data = instance.copy()  # Proxy data can be used as-is for prompts

        evaluation_metadata = {
            "id": instance.get("id", ""),
        }

        return ProcessedInstance(
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )


def get_instance_processor(benchmark_config: BenchmarkConfig) -> InstanceProcessor:
    """Get the appropriate instance processor for the given dataset."""
    if "gpqa" in benchmark_config.dataset_name.lower():
        return GPQAInstanceProcessor()
    elif "swe" in benchmark_config.dataset_name.lower():
        return SWEInstanceProcessor()
    else:
        # Default to proxy processor for unknown datasets
        return ProxyInstanceProcessor()
