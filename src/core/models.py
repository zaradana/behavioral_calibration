from dataclasses import fields
from typing import List

from .schema import ModelConfig

MODELS = [
    ModelConfig(
        model_name="o1", model_path="openai/o1", accepts_image=False, is_free=False
    ),
    ModelConfig(
        model_name="gpt-4o",
        model_path="openai/gpt-4o",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="gpt-4o-mini",
        model_path="openai/gpt-4o-mini",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="deepseek-r1",
        model_path="deepseek/deepseek-r1",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="claude-3.5-sonnet",
        model_path="anthropic/claude-3.5-sonnet",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="claude-3.7-sonnet",
        model_path="anthropic/claude-3.7-sonnet",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="o3-mini",
        model_path="openai/o3-mini",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="gpt-5",
        model_path="openai/gpt-5",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="gemini-2.5-pro",
        model_path="google/gemini-2.5-pro",
        accepts_image=True,
        is_free=False,
    ),
    ModelConfig(
        model_name="llama-3.1-405b-instruct",
        model_path="meta-llama/llama-3.1-405b-instruct:free",
        accepts_image=False,
        is_free=True,
    ),
    ModelConfig(
        model_name="sonoma-sky-alpha",
        model_path="openrouter/sonoma-sky-alpha",
        accepts_image=True,
        is_free=True,
    ),
    ModelConfig(
        model_name="deepseek-chat-v3.1",
        model_path="deepseek/deepseek-chat-v3.1:free",
        accepts_image=False,
        is_free=True,
    ),
    ModelConfig(
        model_name="gemini-2.0-flash-001",
        model_path="google/gemini-2.0-flash-001",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="qwen3-235b-a22b-2507",
        model_path="qwen/qwen3-235b-a22b-2507",
        accepts_image=False,
        is_free=False,
    ),
    ModelConfig(
        model_name="llama-3.2-3b-instruct",
        model_path="meta-llama/llama-3.2-3b-instruct",
        accepts_image=False,
        is_free=False,
    ),
]


def get_model_by_field(field: str = None, field_value: str = None) -> List[ModelConfig]:
    if not field:
        return MODELS

    # Check if filter is a valid ModelConfig field by checking dataclass fields
    valid_fields = {field.name for field in fields(ModelConfig)}
    if field not in valid_fields:
        raise ValueError(
            f"Invalid filter: {field}. Must be an attribute of ModelConfig. Valid fields: {valid_fields}"
        )

    return [model for model in MODELS if str(getattr(model, field)) == str(field_value)]


def get_model_by_name(model_name: str) -> ModelConfig:
    return next((model for model in MODELS if model.model_name == model_name), None)


def get_all_models() -> List[ModelConfig]:
    return MODELS
