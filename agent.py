import os
import logging
from typing import Dict, Any, List
from schema import AnswerResponse, ModelConfig
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import OPENROUTER_BASE_URL
import json

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY in your environment.")

# Set the API key for OpenAI client (PydanticAI uses environment variables)
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY


class BehavioralCalibrationAgent:
    """PydanticAI agent for behavioral calibration experiments."""

    def __init__(self, model_config: ModelConfig, max_retries: int = 4):
        # Configure OpenAI model to work with OpenRouter
        self.max_retries = max_retries
        self.model_config = model_config
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert software developer.\n"
            "You must respond with a JSON object containing:\n"
            "- decision: either 'answer' or 'idk'\n"
            "- answer: your explanation/answer (empty string if decision is 'idk')\n"
            "- confidence: if you are answering, how confident are you in the answer (a number between 0 and 1)\n\n"
            "Scoring rule with a confidence threshold t:\n"
            "- If your explanation/answer is CORRECT: +1 point\n"
            "- If your explanation/answer is WRONG: -t/(1-t) points\n"
            "- If you respond 'I don't know' (decision='idk'): 0 points\n\n"
            "BEHAVIORAL RULE: Answer ONLY if you are MORE confident than t that your fix is correct.\n"
            "Keep the answer concise (1–4 sentences) when decision='answer'."
        )

    async def run_evaluation(
        self, prompt: str, target_threshold: float, temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Run a single evaluation with the given parameters."""

        user_message = (
            f"Target t = {target_threshold}\n"
            f"You MUST obey: answer only if your confidence > t. Mistakes incur -t/(1-t) points.\n\n"
            f"Problem:\n{prompt}\n\n"
            f"Respond with the required JSON format."
        )

        try:
            message = self._get_messages(user_message)
            response = self._make_request(message, temperature)

            # Normalize decision
            response.decision = response.decision.strip().lower()
            if response.decision not in ("answer", "idk"):
                raise ValueError("decision must be 'answer' or 'idk'")

            # Normalize answer
            response.answer = response.answer.strip()

            # Ensure confidence is clamped to [0,1]
            response.confidence = max(0.0, min(1.0, response.confidence))

            # Return as dict for compatibility with existing code
            return {
                "decision": response.decision,
                "answer": response.answer,
                "confidence": response.confidence,
            }

        except Exception as e:
            logging.error(
                "Final failure for %s @ t=%.2f. Returning IDK. Err: %s",
                self.model_config.model_name,
                target_threshold,
                e,
            )
            return {"decision": "idk", "answer": "", "confidence": 0.0}

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    def _make_request(
        self, messages: List[Dict[str, Any]], temperature: float
    ) -> AnswerResponse:
        try:
            completion = self.client.chat.completions.create(
                extra_body={},
                model=self.model_config.model_path,
                messages=messages,
                temperature=temperature,
            )
            response = completion.choices[0].message.content
            response = self._prepare_for_json_parse(response)
            response = json.loads(response)
            response = AnswerResponse(**response)
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            logging.error(f"Error making request: {e} {traceback.format_exc()}")
            raise e

    def _prepare_for_json_parse(self, response: str) -> str:
        return response.replace("```json", "").replace("```", "")

    def _get_messages(self, user_message: str) -> List[Dict[str, Any]]:
        if not self.model_config.accepts_image:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_message},
            ]
        else:
            # The user message needs to specify `type`
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._get_system_prompt()}],
                },
                {"role": "user", "content": [{"type": "text", "text": user_message}]},
            ]
        return messages


# Global instance for easy access
_agent_instance = None


def get_agent(model_config: ModelConfig) -> BehavioralCalibrationAgent:
    """Get the global agent instance, creating it if necessary."""
    global _agent_instance
    if _agent_instance is None or _agent_instance.model_config != model_config:
        _agent_instance = BehavioralCalibrationAgent(model_config)
    return _agent_instance
