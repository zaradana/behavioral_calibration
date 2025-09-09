import os

from dotenv import load_dotenv

from schema import ModelConfig

load_dotenv()

OUTPUT_DIR = "./output"
LOGS_DIR = "./logs"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY in your environment.")

BENCHMARK_DATASET = "SWE-bench/SWE-bench_Lite"
DEV_MODE = True

TARGETS = [0.5 , 0.7, 0.9]

# Check model IDs against your OpenRouter catalog
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
        model_name="nemotron-nano-9b-v2",
        model_path="nvidia/nemotron-nano-9b-v2",
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
]
