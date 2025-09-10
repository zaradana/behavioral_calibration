from swebench.inference.make_datasets.create_instance import make_code_text, PATCH_EXAMPLE
from typing import Dict

def prompt_style_2(instance: Dict[str, str], behavioral_calibration_prompt: str) -> str:
    # Reproduced from https://github.com/SWE-bench/SWE-bench/tree/db2f39a3a6b808c70ff8f3f67ae7d0dd48829cff/swebench/inference/make_datasets/create_instance.py
    # with small modifications to include the behavioral calibration prompt

    premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
    readmes_text = make_code_text(instance.get("readmes", {}))
    code_text = make_code_text(instance.get("file_contents", {}))
    instructions = (
        "I need you to solve this issue by generating a single patch file that I can apply "
        + "directly to this repository using git apply. Please respond with a single patch "
        + "file in the following format."
    )
    instructions += behavioral_calibration_prompt
    problem_statement = instance.get("problem_statement", "")
    final_text = [
        premise,
        "<issue>",
        problem_statement,
        "</issue>",
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        instructions,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
    ]
    final_text = "\n".join(final_text)
    return final_text
