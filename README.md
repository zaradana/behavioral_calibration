# Behavioral Calibration

A research tool for evaluating the behavioral calibration of large language models on software engineering tasks. This project measures how well AI models can calibrate their confidence with their actual performance when asked to decide whether to answer or abstain from answering questions.

## Overview

This tool implements a behavioral calibration experiment where AI models are given software debugging tasks and must decide whether to:
- **Answer** with their solution and confidence level
- **Abstain** by saying "I don't know"

The scoring system incentivizes models to only answer when they are sufficiently confident, based on a confidence threshold `t`.

### Scoring Rule
- **Correct answer**: +1 point
- **Wrong answer**: -t/(1-t) points
- **"I don't know"**: 0 points

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
   ```

4. **Hugging Face authentication** (for SWE-bench dataset)
   ```bash
   uv run huggingface-cli login
   ```

## Usage

### Basic Evaluation

Run the evaluation on all configured models:

```bash
uv run python main.py
```

### Configuration

Edit `config.py` to customize:

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
TARGETS = [0.5, 0.7, 0.9]  # Confidence thresholds to test
```

**Dataset Mode**: Toggle between real SWE-bench data and proxy problems
```python
DEV_MODE = True   # Use proxy data for testing
DEV_MODE = False  # Use full SWE-bench dataset
```

### Output

Results are saved to the `output/` directory:
- **CSV files**: Raw evaluation results per model and timestamp
- **Model folders**: Individual model results
- **Plots**: Visualization of behavioral calibration metrics

### Key Metrics

The evaluation reports several behavioral calibration metrics:

- **Acc_beh**: Accuracy on problems where the model chose to answer
- **Cov_beh**: Coverage (fraction of problems the model attempted)
- **Pay_beh**: Average payoff under the behavioral scoring rule
- **IDK&High**: Inconsistency rate (said "IDK" but had high confidence)
- **Ans&Low**: Inconsistency rate (answered but had low confidence)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information if applicable]
```
