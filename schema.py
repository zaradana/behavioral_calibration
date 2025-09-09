from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Literal, Dict, Any


class AnswerResponse(BaseModel):
    """Response model for SWE diagnostic evaluations."""

    decision: Literal["answer", "idk"] = Field(
        ..., description="Whether to provide an answer or say 'I don't know'"
    )
    answer: str = Field(
        ..., description="The diagnostic answer or fix (empty if decision is 'idk')"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level between 0 and 1"
    )


@dataclass
class ItemEval:
    item_id: str
    decision: str  # "answer" or "idk"
    answer: str
    confidence: float
    correct: bool
    payoff_behavioral: float  # payoff using decision as the abstention mechanism


@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    accepts_image: bool
    is_free: bool
