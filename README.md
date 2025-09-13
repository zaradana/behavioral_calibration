# Behavioral Calibration

A research tool for evaluating the behavioral calibration of large language models across multiple benchmarks. This project measures how well AI models can calibrate their confidence with their actual performance when asked to decide whether to answer or abstain from answering questions.

## Overview

This tool implements a behavioral calibration experiment where AI models are given tasks from various benchmarks and must decide whether to:
- **Answer** with their solution and confidence level
- **Abstain** by saying "I don't know"

The system supports multiple benchmark types, each with specialized evaluation logic.

The scoring system incentivizes models to only answer when they are sufficiently confident, based on a confidence threshold `t`.

### Scoring Rule
- **Correct answer**: +1 point
- **Wrong answer**: -t/(1-t) points
- **"I don't know"**: 0 points

## Features

- **Multi-benchmark Support**: Evaluate across SWE-bench, GPQA, and proxy tasks
- **Multi-model evaluation**: Test different LLMs from OpenRouter
- **Behavioral analysis**: Measure how well models follow confidence-based abstention rules
- **Configurable thresholds**: Test different confidence thresholds
- **Comprehensive metrics**: Coverage, accuracy, payoff, and consistency rates
- **Visualization**: Generate plots showing model behavior across thresholds
- **Extensible Architecture**: Easy to add new benchmark types

### Adding New Benchmarks

The modular pattern makes it easy to add new benchmarks:
1. **Instance processor**: Create an instance processor in `src/utils/instance_processor.py` to process the instance for the prompt, and evaluation.
2. **Create evaluator**: Implement `BaseBenchmarkEvaluator` in `src/evaluations/`
3. **Register evaluator**: Add to `EvaluatorFactory`
4. **Add prompts**: Create prompt templates in `src/prompts/`
5. **Add prompt factory**: Add prompt factory in `src/prompts/prompt_factory.py`
6. **Configure**: Add benchmark config to `src/core/benchmarks.py`


### Adding New Models
1. **Find model**: Find the model in the [OpenRouter](https://openrouter.ai/models) and get the model ID.
2. **Create model config**: Create a model config in `src/core/models.py` and add it to the `MODELS` list.
```python
ModelConfig(
    model_name="your-model",
    model_path="provider/model-id",
    accepts_image=False,
    is_free=True  # Set to False for paid models
)
```

## Supported Benchmarks

### SWE-bench
- **Task**: Software engineering bug fixes
- **Format**: Generate code patches to fix real GitHub issues
- **Evaluation**: Patches tested against actual repository test suites using Docker
- **Metrics**: Pass/fail based on test execution

### GPQA
- **Task**: Graduate-level physics questions
- **Format**: Multiple choice (A, B, C, D)
- **Evaluation**: Exact match against correct answer
- **Metrics**: Binary correctness

### Proxy Tasks
- **Task**: Simplified debugging problems
- **Format**: String matching against expected outputs
- **Evaluation**: Heuristic substring matching
- **Metrics**: Binary correctness based on expected substrings

### GSM8K
- **Task**: Math problems
- **Format**: Textual answer
- **Evaluation**: Exact match against correct answer
- **Metrics**: Binary correctness

### SVAMP
- **Task**: Math problems
- **Format**: Textual answer
- **Evaluation**: Exact match against correct answer
- **Metrics**: Binary correctness

## Installation

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/zaradana/behavioral_calibration.git
   cd behavioral_calibration
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Environment configuration**
   Create a `.env` file in the project root:
   ```bash
   # Required: OpenRouter API key for model access
   OPENROUTER_API_KEY=your_api_key_here
   HF_HUB_TOKEN=your_token_here
   ENV=dev
   ```
   Note: `HF_HUB_TOKEN` is only need for the gated benchmarks like GPQA.


   **For macOS users with Docker Desktop**: The Docker socket path is automatically configured. If you encounter Docker connection issues, ensure Docker Desktop is running.

   **For Linux users**: Ensure your user is in the `docker` group and Docker daemon is running.



## Usage

### Basic Evaluation

Run the evaluation using the default configuration:

```bash
uv run python -m src.cli.measure_calibration
```

### Command Line Options

You can override the default configuration using command line arguments:

```bash
# Use specific models (overrides config)
uv run python -m src.cli.measure_calibration --models gpt-4o claude-3.5-sonnet

# Use specific benchmark (overrides config)
uv run python -m src.cli.measure_calibration --benchmark gsm8k

# Combine both
uv run python -m src.cli.measure_calibration --models gpt-4o-mini --benchmark gpqa

# See all available options
uv run python -m src.cli.measure_calibration --help
```

**Available Models:** Run `--help` to see the current list of supported models

**Available Benchmarks:** `swe_bench_lite`, `proxy_data`, `gpqa`, `truthfulqa`, `gsm8k`, `svamp`

### Configuration

Edit `src/core/config.py` to customize:

**Models**: Add/remove models from the `MODELS` list
```python
ModelConfig(
    model_name="your-model",
    model_path="provider/model-id",
    accepts_image=False,
    is_free=True  # Set to False for paid models
)
```

**Confidence Thresholds**: Modify the `TARGETS` list
```python
CONFIDENCE_TARGETS = [0.0, 0.5, 0.8]  # Confidence thresholds to test
```

**Benchmark Selection**: Choose which benchmarks to evaluate
```python
# Configure benchmark in src/core/config.py
BENCHMARK = get_benchmark_config("gpqa") # or "swe_bench_lite", "proxy_data"
```

**Environment Mode**: Changes the logging level.
```python
ENV = "dev"   # Use DEBUG level logging
ENV = "prod"  # Use INFO level logging
```

### Output

Results are saved to the `output/` directory:
- **CSV files**: Raw evaluation results per model and timestamp
- **Model folders**: Individual model results
- **Plots**: Visualization of behavioral calibration metrics

### Key Metrics

The evaluation reports several behavioral calibration metrics across all supported benchmarks:

#### Behavioral Metrics
- **Acc_beh**: Accuracy on problems where the model chose to answer
- **Cov_beh**: Coverage (fraction of problems the model attempted)
- **Pay_beh**: Average payoff under the behavioral scoring rule
- **IDK&High**: Inconsistency rate (said "IDK" but had high confidence)
- **Ans&Low**: Inconsistency rate (answered but had low confidence)

#### Benchmark-Specific Metrics
- **SWE-bench**: Test pass/fail rates, patch application success
- **GPQA**: Multiple-choice accuracy, answer extraction success
- **Proxy**: Substring matching accuracy, heuristic evaluation

All metrics are computed consistently across benchmarks using the evaluator pattern.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information if applicable]
```
