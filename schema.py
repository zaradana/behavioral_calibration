from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """Response model for SWE diagnostic evaluations."""

    decision: Literal["answer", "idk"] = Field(
        default="idk", description="Whether to provide an answer or say 'I don't know'"
    )
    answer: str = Field(
        default="", description="The answer (empty if decision is 'idk')"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence level between 0 and 1"
    )
    evaluation_metadata: Optional[Dict] = Field(
        default=None, description="Metadata needed for evaluation (e.g., correct_index)"
    )


@dataclass
class ItemEval:
    decision: str  # "answer" or "idk"
    answer: str
    confidence: float
    correct: Optional[bool] = None
    payoff_behavioral: Optional[float] = (
        None  # payoff using decision as the abstention mechanism
    )
    evaluation_metadata: Optional[Dict] = None  # Store processed instance metadata


@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    accepts_image: bool
    is_free: bool


# Benchmark configuration dataclass for dynamic dataset selection
@dataclass
class BenchmarkConfig:
    dataset_name: str
    dataset_path: str
    config_name: str
    dataset_split: str = "test"


class ConfigSchema(BaseModel):
    TEMPERATURE: float
    BENCHMARK: BenchmarkConfig
    CONFIDENCE_TARGETS: List[float]
    MODELS: List[ModelConfig]


@dataclass
class SWEBenchPrediction:
    """SWE-bench prediction format"""

    instance_id: str
    model_patch: str
    model_name_or_path: str


@dataclass
class SWEBenchResult:
    """SWE-bench evaluation result for a single instance"""

    instance_id: str
    resolved: bool
    error_message: str = ""
    test_output: str = ""
    patch_applied: bool = False


@dataclass
class SWEBenchSummary:
    """Summary of SWE-bench evaluation results"""

    total_instances: int
    instances_submitted: int
    instances_completed: int
    instances_resolved: int
    resolution_rate: float  # resolved / submitted
    completion_rate: float  # completed / submitted


@dataclass
class ProcessedInstance:
    """Container for processed instance data with metadata."""

    original_instance: Dict[str, Any]
    prompt_data: Dict[str, Any]  # Data needed for prompt generation
    evaluation_metadata: Dict[
        str, Any
    ]  # Data needed for evaluation (e.g., correct_index)
