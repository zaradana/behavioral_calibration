import random
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.schema import BenchmarkConfig, ProcessedInstance
from .core_utils import get_logger

logger = get_logger(__name__)


class InstanceProcessor(ABC):
    """Base class for processing benchmark instances."""

    def __init__(self):
        """Initialize the instance processor with its own ID counter."""
        # Each instance gets its own counter starting from 1
        self.id = 0

    def _get_next_id(self) -> int:
        """Get the next ID using this instance's atomic increment."""
        self.id += 1
        logger.info(f"Instance ID: {self.id}")
        return self.id

    @abstractmethod
    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process a raw instance into prompt data and evaluation metadata."""
        pass


class GPQAInstanceProcessor(InstanceProcessor):
    """Processor for GPQA instances that handles answer shuffling."""

    def __init__(self):
        super().__init__()

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process GPQA instance by shuffling answers and tracking correct index."""
        # Shuffle the answer choices
        example_choices = [
            instance["Incorrect Answer 1"],
            instance["Incorrect Answer 2"],
            instance["Incorrect Answer 3"],
            instance["Correct Answer"],
        ]

        # Use current timestamp + instance-specific data as seed for true randomness
        seed = hash(str(instance))
        random.seed(seed)
        random.shuffle(example_choices)

        # Find the correct answer index after shuffling
        correct_index = example_choices.index(instance["Correct Answer"])

        prompt_data = {
            "question": instance["Question"],
            "choice1": example_choices[0],
            "choice2": example_choices[1],
            "choice3": example_choices[2],
            "choice4": example_choices[3],
        }

        evaluation_metadata = {"correct_index": correct_index}

        # Create ProcessedInstance with placeholder ID first to avoid blocking
        result = ProcessedInstance(
            id=0,  # Temporary placeholder
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )
        # Assign the actual ID after creation to minimize lock contention
        result.id = self._get_next_id()
        return result


class SWEInstanceProcessor(InstanceProcessor):
    """Processor for SWE-bench instances."""

    def __init__(self):
        super().__init__()

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

        # Create ProcessedInstance with placeholder ID first to avoid blocking
        result = ProcessedInstance(
            id=0,  # Temporary placeholder
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )
        # Assign the actual ID after creation to minimize lock contention
        result.id = self._get_next_id()
        return result


class ProxyInstanceProcessor(InstanceProcessor):
    """Processor for proxy data instances."""

    def __init__(self):
        super().__init__()

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process proxy instance - mostly pass-through."""
        prompt_data = instance.copy()  # Proxy data can be used as-is for prompts

        evaluation_metadata = {
            "expect_substrings": instance.get("expect_substrings", []),
        }

        # Create ProcessedInstance with placeholder ID first to avoid blocking
        result = ProcessedInstance(
            id=0,  # Temporary placeholder
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )
        # Assign the actual ID after creation to minimize lock contention
        result.id = self._get_next_id()
        return result


class TruthfulQAInstanceProcessor(InstanceProcessor):
    """Processor for TruthfulQA instances."""

    def __init__(self):
        super().__init__()

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process TruthfulQA instance."""
        prompt_data = {
            "question": instance.get("question", ""),
            "choice1": instance["choices"][0],
            "choice2": instance["choices"][1],
            "choice3": instance["choices"][2],
            "choice4": instance["choices"][3],
        }
        evaluation_metadata = {
            "correct_index": instance.get("label", ""),
        }

        # Create ProcessedInstance with placeholder ID first to avoid blocking
        result = ProcessedInstance(
            id=0,  # Temporary placeholder
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )
        # Assign the actual ID after creation to minimize lock contention
        result.id = self._get_next_id()
        return result


class GSM8KInstanceProcessor(InstanceProcessor):
    """Processor for GSM8K instances."""

    def __init__(self):
        super().__init__()

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process GSM8K instance to extract problem and answer."""
        question = instance.get("question", "")
        answer = instance.get("answer", "")

        # Extract the final numeric answer from the solution
        # GSM8K answers are in format like "#### 42"
        numeric_answer = ""
        if "####" in answer:
            numeric_answer = answer.split("####")[-1].strip()

        prompt_data = {"problem": question}

        evaluation_metadata = {
            "answer": numeric_answer,
        }

        # Create ProcessedInstance with placeholder ID first to avoid blocking
        result = ProcessedInstance(
            id=0,  # Temporary placeholder
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )
        # Assign the actual ID after creation to minimize lock contention
        result.id = self._get_next_id()
        return result


class SVAMPInstanceProcessor(InstanceProcessor):
    """Processor for SVAMP instances."""

    def __init__(self):
        super().__init__()

    def process(self, instance: Dict[str, Any]) -> ProcessedInstance:
        """Process SVAMP instance using question_concat and Answer columns."""
        # SVAMP columns: ID, Body, Question, Equation, Answer, Type, question_concat
        question_concat = instance.get("question_concat", "")
        answer = instance.get("Answer", "")
        prompt_data = {"question_concat": question_concat}

        evaluation_metadata = {
            "answer": answer,
        }

        # Create ProcessedInstance with placeholder ID first to avoid blocking
        result = ProcessedInstance(
            id=0,  # Temporary placeholder
            original_instance=instance,
            prompt_data=prompt_data,
            evaluation_metadata=evaluation_metadata,
        )
        # Assign the actual ID after creation to minimize lock contention
        result.id = self._get_next_id()
        return result


def get_instance_processor(benchmark_config: BenchmarkConfig) -> InstanceProcessor:
    """Get a new instance processor for the given dataset.

    Returns a new instance each time so each agent gets its own processor
    with independent ID counting starting from 1.
    """
    dataset_key = benchmark_config.dataset_name.lower()

    # Create new processor instance each time
    if "gpqa" in dataset_key:
        processor = GPQAInstanceProcessor()
    elif "swe" in dataset_key:
        processor = SWEInstanceProcessor()
    elif "truthfulqa" in dataset_key:
        processor = TruthfulQAInstanceProcessor()
    elif "gsm8k" in dataset_key:
        processor = GSM8KInstanceProcessor()
    elif "svamp" in dataset_key:
        processor = SVAMPInstanceProcessor()
    else:
        # Default to proxy processor for unknown datasets
        processor = ProxyInstanceProcessor()

    return processor
