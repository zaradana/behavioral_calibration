import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from ..prompts.prompt_factory import PromptFactory
from ..utils.instance_processor import get_instance_processor
from .config import OPENROUTER_BASE_URL, TEMPERATURE
from .schema import AnswerResponse, BenchmarkConfig, ModelConfig, SimpleAnswerResponse

load_dotenv()


def is_rate_limit_error(exception):
    """Check if exception is a rate limiting error."""
    if exception is None:
        return False

    try:
        error_str = str(exception).lower()

        # Check common rate limiting indicators
        rate_limit_indicators = [
            "rate_limit",
            "rate limit",
            "ratelimit",
            "too many requests",
            "quota",
            "throttl",
        ]

        # Check error message
        for indicator in rate_limit_indicators:
            if indicator in error_str:
                return True

        # Check status codes
        if "429" in str(exception):
            return True

        # Check status_code attribute (OpenAI and similar APIs)
        if hasattr(exception, "status_code") and exception.status_code == 429:
            return True

        # Check response attribute (some APIs nest the status code)
        if hasattr(exception, "response") and hasattr(
            exception.response, "status_code"
        ):
            if exception.response.status_code == 429:
                return True

        # Check for OpenAI specific rate limit exceptions
        if exception.__class__.__name__ in ["RateLimitError", "APIError"]:
            return True

        return False

    except Exception:
        # If anything goes wrong in detection, assume it's not rate limiting
        # This prevents the detection logic itself from breaking
        return False


def should_retry_rate_limiting(retry_state):
    """Custom retry condition: only retry on rate limiting errors."""
    if not retry_state.outcome or not retry_state.outcome.failed:
        # This shouldn't happen - tenacity only calls this after failures
        return False

    exception = retry_state.outcome.exception()
    is_rate_limit = is_rate_limit_error(exception)
    logging.info(
        f"Rate limiting strategy: exception={exception}, is_rate_limit={is_rate_limit}"
    )
    return is_rate_limit


def should_retry_general(retry_state):
    """Custom retry condition: only retry on non-rate-limiting errors."""
    if not retry_state.outcome or not retry_state.outcome.failed:
        # This shouldn't happen - tenacity only calls this after failures
        return False

    exception = retry_state.outcome.exception()
    is_rate_limit = is_rate_limit_error(exception)
    logging.info(
        f"General strategy: exception={exception}, is_rate_limit={is_rate_limit}"
    )
    return not is_rate_limit


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY in your environment.")

# Set the API key for OpenAI client (PydanticAI uses environment variables)
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY


class BehavioralCalibrationAgent:
    """PydanticAI agent for behavioral calibration experiments."""

    def __init__(
        self,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        max_retries: int = 4,
    ):
        # Configure OpenAI model to work with OpenRouter
        self.max_retries = max_retries
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        # Create a persistent processor for this agent instance
        self.processor = get_instance_processor(benchmark_config)

    async def run(
        self,
        instance: Dict[str, str],
        target_threshold: float,
    ) -> AnswerResponse:
        """Run a single evaluation with the given parameters."""

        # Process the instance to extract prompt data and evaluation metadata
        processed_instance = self.processor.process(instance)

        user_message = PromptFactory.get_prompt(
            self.benchmark_config, processed_instance, target_threshold
        )

        try:
            message = self._get_messages(user_message)
            response = await self._make_request(message, TEMPERATURE, target_threshold)

            # Normalize decision
            response.decision = response.decision.strip().lower()
            if response.decision not in ("answer", "idk"):
                raise ValueError("decision must be 'answer' or 'idk'")

            # Normalize answer
            response.answer = response.answer.strip()

            # Add evaluation metadata to the response for later use
            response.evaluation_metadata = processed_instance.evaluation_metadata
            response.id = processed_instance.id

            # Return as dict for compatibility with existing code
            return response

        except Exception as e:
            logging.error(
                "Final failure for %s @ t=%.2f. Returning empty answer. Err: %s",
                self.model_config.model_name,
                target_threshold,
                e,
            )
            return AnswerResponse(
                id=processed_instance.id,
                decision="answer",  # the idk should be reserved for when the model is sure that it doesn't know the answer
                answer="",
                confidence=0.0,
                evaluation_metadata=processed_instance.evaluation_metadata,
            )

    async def _make_single_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        target_threshold: float,
    ) -> AnswerResponse:
        """Make a single API request without any retry logic."""
        completion = await self.client.chat.completions.create(
            extra_body={},
            model=self.model_config.model_path,
            messages=messages,
            temperature=temperature,
        )
        response = completion.choices[0].message.content
        logging.debug(f"Agent Response: {response}")

        try:
            cleaned_response = self._prepare_for_json_parse(response)
            parsed_json = json.loads(cleaned_response)

            # Convert answer to string if it's not already (models sometimes return numbers)
            if "answer" in parsed_json and not isinstance(parsed_json["answer"], str):
                parsed_json["answer"] = str(parsed_json["answer"])
                logging.info(
                    f"Converted answer from {type(parsed_json['answer'])} to string"
                )

            if target_threshold == 0.0:
                simple_response = SimpleAnswerResponse(**parsed_json)
                answer_response = AnswerResponse(
                    decision="answer",
                    answer=simple_response.answer,
                    confidence=simple_response.confidence,
                    evaluation_metadata=simple_response.evaluation_metadata,
                    id=simple_response.id,
                )
            else:
                answer_response = AnswerResponse(**parsed_json)

            return answer_response
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}")
            logging.error(
                f"Cleaned response that failed to parse: {repr(cleaned_response)}"
            )
            raise
        except Exception as e:
            logging.error(f"AnswerResponse creation failed: {e}")
            logging.error(f"Parsed JSON: {parsed_json}")
            raise

    @retry(
        # For general API errors: shorter exponential backoff (2s, 4s, 8s)
        # Retry on any exception EXCEPT rate limiting (those should use rate limiting strategy)
        retry=should_retry_general,
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=2, min=2, max=16
        ),  # 2s, 4s, 8s (capped at 16s)
        before_sleep=before_sleep_log(logging, logging.WARNING),
    )
    async def _make_general_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        target_threshold: float,
    ) -> AnswerResponse:
        """Make API request with general error handling - used after first attempt fails with non-rate-limit error."""
        return await self._make_single_request(messages, temperature, target_threshold)

    @retry(
        # For rate limiting errors: longer exponential backoff (5s, 10s, 20s, 40s)
        # Only retry on rate limiting errors - other errors should not use this strategy
        retry=should_retry_rate_limiting,
        stop=stop_after_attempt(4),
        wait=wait_exponential(
            multiplier=5, min=5, max=60
        ),  # 5s, 10s, 20s, 40s (capped at 60s)
        before_sleep=before_sleep_log(logging, logging.WARNING),
    )
    async def _make_rate_limited_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        target_threshold: float,
    ) -> AnswerResponse:
        """Make API request with longer backoff for rate limiting - tenacity handles retries."""
        return await self._make_single_request(messages, temperature, target_threshold)

    async def _make_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        target_threshold: float,
    ) -> AnswerResponse:
        """Make API request with intelligent retry strategy that adapts to error types."""

        # First attempt - no retry, just detect error type
        try:
            return await self._make_single_request(
                messages, temperature, target_threshold
            )
        except Exception as first_error:
            if is_rate_limit_error(first_error):
                # Rate limited on first try → use rate limiting strategy
                logging.info(
                    "Rate limiting detected on first attempt, using rate limiting strategy"
                )
                try:
                    return await self._make_rate_limited_request(
                        messages, temperature, target_threshold
                    )
                except Exception as rate_limit_strategy_error:
                    # If rate limiting strategy encounters non-rate-limit error, try general strategy
                    if not is_rate_limit_error(rate_limit_strategy_error):
                        logging.info(
                            "Rate limiting strategy encountered non-rate-limit error, switching to general strategy"
                        )
                        return await self._make_general_request(
                            messages, temperature, target_threshold
                        )
                    else:
                        # Still rate limiting error after rate limiting strategy
                        logging.error(
                            f"Rate limiting persists after all retries {rate_limit_strategy_error}."
                        )
                        raise rate_limit_strategy_error
            else:
                # Other error on first try → use general strategy
                logging.info(
                    f"General error detected on first attempt ({first_error}), using general strategy"
                )
                try:
                    return await self._make_general_request(
                        messages, temperature, target_threshold
                    )
                except Exception as general_strategy_error:
                    # If general strategy encounters rate-limit error, try rate limiting strategy
                    if is_rate_limit_error(general_strategy_error):
                        logging.info(
                            "General strategy encountered rate limiting, switching to rate limiting strategy"
                        )
                        return await self._make_rate_limited_request(
                            messages, temperature, target_threshold
                        )
                    else:
                        # Still general error after general strategy∂
                        logging.error(
                            f"General errors persist after all retries {general_strategy_error}."
                        )
                        raise general_strategy_error

    def _prepare_for_json_parse(self, response: str) -> str:
        """Clean and prepare model response for JSON parsing."""

        # Remove any code block markers
        response = response.replace("```json", "").replace("```", "").strip()

        # Find the first complete JSON object
        try:
            start = response.index("{")
            brace_count = 0
            end = start

            for i in range(start, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            json_part = response[start:end]
            logging.debug(f"Extracted JSON: {json_part}")
            return json_part
        except ValueError as e:
            # If no JSON object found, log and return original cleaned response
            logging.warning(f"No JSON object found in response. ValueError: {e}")
            logging.warning(f"Response content: {repr(response)}")
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


def get_agent(
    model_config: ModelConfig, benchmark_config: BenchmarkConfig
) -> BehavioralCalibrationAgent:
    """Get the global agent instance, creating it if necessary."""
    global _agent_instance
    if _agent_instance is None or _agent_instance.model_config != model_config:
        _agent_instance = BehavioralCalibrationAgent(model_config, benchmark_config)
    return _agent_instance
