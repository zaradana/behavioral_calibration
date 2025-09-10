import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENROUTER_BASE_URL, TEMPERATURE
from prompts.prompt_factory import PromptFactory
from schema import AnswerResponse, BenchmarkConfig, ModelConfig
from utils.instance_processor import get_instance_processor

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

    async def run(
        self,
        instance: Dict[str, str],
        target_threshold: float,
        benchmark_config: BenchmarkConfig,
    ) -> AnswerResponse:
        """Run a single evaluation with the given parameters."""
        
        # Process the instance to extract prompt data and evaluation metadata
        processor = get_instance_processor(benchmark_config)
        processed_instance = processor.process(instance)
    
        user_message = PromptFactory.get_prompt(
            benchmark_config, processed_instance, target_threshold
        )

        try:
            message = self._get_messages(user_message)
            response = self._make_request(message, TEMPERATURE)

            print(f"Response: {response}")

            # Normalize decision
            response.decision = response.decision.strip().lower()
            if response.decision not in ("answer", "idk"):
                raise ValueError("decision must be 'answer' or 'idk'")

            # Normalize answer
            response.answer = response.answer.strip()

            # Add evaluation metadata to the response for later use
            response.evaluation_metadata = processed_instance.evaluation_metadata

            # Return as dict for compatibility with existing code
            return response

        except Exception as e:
            logging.error(
                "Final failure for %s @ t=%.2f. Returning IDK. Err: %s",
                self.model_config.model_name,
                target_threshold,
                e,
            )
            return AnswerResponse(decision="idk", answer="", confidence=0.0)

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
            logging.debug(f"Agent Response: {response}")
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
        """Clean and prepare model response for JSON parsing."""
        # Remove any code block markers
        response = response.replace("```json", "").replace("```", "").strip()

        # Remove any non-JSON text before/after the JSON object
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            response = response[start:end]
        except ValueError:
            # If no JSON object found, return original cleaned response
            pass

        return response

    def _get_messages(self, user_message: str) -> List[Dict[str, Any]]:
        if not self.model_config.accepts_image:
            messages = [
                {"role": "user", "content": user_message},
            ]
        else:
            # The user message needs to specify `type`
            messages = [
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
