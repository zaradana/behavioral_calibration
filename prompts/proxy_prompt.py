from typing import Dict

def get_proxy_prompt(instance: Dict[str, str], behavioral_calibration_prompt: str = "") -> str:
    return behavioral_calibration_prompt + f"Problem:\n{instance['prompt']}\n\n"
