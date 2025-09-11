import os

from dotenv import load_dotenv

from benchmarks import get_benchmark_config
from models import get_model_by_name

load_dotenv()

# Benchmark configuration - easily configurable
# Option 1: Use environment variables (current approach)
BENCHMARK = get_benchmark_config("gpqa")
OUTPUT_DIR = "./output"
LOGS_DIR = "./logs"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
TEMPERATURE = 0  # deterministic behavior is desired for the calibration experiments
ENV = os.getenv("ENV", "dev")
# Behavioral calibration settings
CONFIDENCE_TARGETS = [0.0, 0.5, 0.8]  # 0 is to be used for simulating IDK
# MODELS = get_model_by_field(field="is_free", field_value=True)  # get all free models
MODELS = [get_model_by_name("sonoma-sky-alpha")]
