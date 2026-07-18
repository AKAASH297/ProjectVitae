import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from project_vitae.config import Config, RetryConfig, SubagentConfig
from project_vitae.cost import CostGuard, compute_cost
from project_vitae.models import (
    LLMCallError,
    LLMCallRecord,
    TokenBudgetExceeded,
)
from project_vitae.prompts import resolve_prompt

logger = logging.getLogger(__name__)

_jsonl_lock = threading.Lock()


class LLMCallResult:
    def __init__(
        self,
        output: BaseModel,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration_ms: int,
        model: str,
        prompt_version: str,
    ):
        self.output = output
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.duration_ms = duration_ms
        self.model = model
        self.prompt_version = prompt_version


class LLMCall:
    def __init__(
        self,
        subagent_name: str,
        cfg: SubagentConfig,
        session_dir: Path,
        output_schema: type[BaseModel],
        cost_guard: CostGuard,
        retry_cfg: RetryConfig | None = None,
        budget_accumulator: list[int] | None = None,
        budget_limit: int | None = None,
        config: Config | None = None,
    ):
        self.subagent_name = subagent_name
        self.cfg = cfg
        self.session_dir = session_dir
        self.output_schema = output_schema
        self.cost_guard = cost_guard
        self.retry_cfg = retry_cfg or RetryConfig()
        self.budget_accumulator = budget_accumulator
        self.budget_limit = budget_limit
        self.config = config

    def invoke(
        self, messages: list[BaseMessage], prompt_override: str | None = None
    ) -> LLMCallResult:
        prompt_version = self.cfg.prompt_version
        system_prompt = prompt_override or resolve_prompt(self.subagent_name, self.cfg)
        full_messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *messages]

        model_instance = self._build_model()

        if self.cfg.provider == "anthropic":
            structured = model_instance.with_structured_output(self.output_schema)
        else:
            parser = PydanticOutputParser(pydantic_object=self.output_schema)
            format_instructions = parser.get_format_instructions()
            full_messages[0] = SystemMessage(content=system_prompt + "\n\n" + format_instructions)

        last_error: Exception | None = None
        for attempt in range(1, self.retry_cfg.max_attempts + 1):
            start = time.monotonic()
            response: BaseModel | None = None
            raw_response = None
            try:
                if self.cfg.provider == "anthropic":
                    raw_response = structured.invoke(full_messages)
                    response = raw_response
                else:
                    raw_response = model_instance.invoke(full_messages)
                    content = (
                        raw_response.content
                        if hasattr(raw_response, "content")
                        else str(raw_response)
                    )
                    response = parser.parse(content)
                duration_ms = int((time.monotonic() - start) * 1000)
            except Exception as e:
                duration_ms = int((time.monotonic() - start) * 1000)
                if self._is_recoverable(e):
                    if attempt < self.retry_cfg.max_attempts:
                        backoff = (
                            self.retry_cfg.backoff_seconds[attempt - 1]
                            if attempt - 1 < len(self.retry_cfg.backoff_seconds)
                            else self.retry_cfg.backoff_seconds[-1]
                        )
                        logger.warning(
                            "LLM call %s attempt %d failed (%s), retrying in %ds",
                            self.subagent_name,
                            attempt,
                            e,
                            backoff,
                        )
                        time.sleep(backoff)
                        last_error = e
                        continue
                    else:
                        raise LLMCallError(f"{self.subagent_name} exhausted retries: {e}") from e
                else:
                    raise LLMCallError(f"{self.subagent_name} non-recoverable error: {e}") from e

            if not isinstance(response, self.output_schema):
                raise LLMCallError(
                    f"expected {self.output_schema.__name__}, got {type(response).__name__}"
                )

            if self.cfg.provider == "anthropic":
                usage = self._extract_usage(response)
            else:
                usage = (
                    self._extract_openai_usage(raw_response)
                    if raw_response
                    else {"input_tokens": 0, "output_tokens": 0}
                )
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            if self.budget_limit is not None and self.budget_accumulator is not None:
                self.budget_accumulator[0] += input_tokens + output_tokens
                if self.budget_accumulator[0] > self.budget_limit:
                    raise TokenBudgetExceeded(
                        f"{self.subagent_name} token budget of {self.budget_limit} "
                        f"exceeded (used: {self.budget_accumulator[0]})"
                    )

            cost = compute_cost(
                self.cfg.model,
                input_tokens,
                output_tokens,
                overrides=self.config.cost.pricing_overrides if self.config else None,
            )
            if self.cost_guard is not None:
                self.cost_guard.spend(cost, was_llm=True)

            record = LLMCallRecord(
                timestamp=datetime.now(timezone.utc),
                subagent=self.subagent_name,
                model=self.cfg.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                duration_ms=duration_ms,
                prompt_version=prompt_version,
            )
            self._append_jsonl(record)

            return LLMCallResult(
                output=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                duration_ms=duration_ms,
                model=self.cfg.model,
                prompt_version=prompt_version,
            )

        raise LLMCallError(
            f"{self.subagent_name} failed after "
            f"{self.retry_cfg.max_attempts} attempts: {last_error}"
        )

    def _build_model(self) -> Any:
        api_key = self.cfg.api_key or (
            os.environ.get(self.cfg.api_key_env) if self.cfg.api_key_env else None
        )
        if not api_key:
            raise LLMCallError(f"API key not set for {self.subagent_name}")
        kwargs = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.provider == "anthropic":
            kwargs["api_key"] = api_key
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            return ChatAnthropic(**kwargs)
        elif self.cfg.provider == "openai_compatible":
            kwargs["api_key"] = api_key
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            else:
                kwargs["base_url"] = "https://api.openai.com/v1"
            return ChatOpenAI(**kwargs)
        else:
            raise LLMCallError(f"unknown provider: {self.cfg.provider}")

    def _is_recoverable(self, error: Exception) -> bool:
        err_str = str(error).lower()
        class_name = type(error).__name__
        if class_name in (
            "RateLimitError",
            "InternalServerError",
            "APIConnectionError",
            "ServiceUnavailableError",
        ):
            return True
        if "429" in err_str or "503" in err_str or "502" in err_str or "504" in err_str:
            return True
        if "rate limit" in err_str or "too many requests" in err_str:
            return True
        return False

    def _extract_usage(self, response: Any) -> dict[str, int]:
        try:
            raw = getattr(response, "response_metadata", {}) or {}
            usage = raw.get("usage", {}) or {}
            if isinstance(usage, dict):
                return {
                    "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
                    "output_tokens": usage.get("output_tokens")
                    or usage.get("completion_tokens")
                    or 0,
                }
        except Exception:
            pass
        try:
            if hasattr(response, "usage_metadata"):
                um = response.usage_metadata
                if um:
                    return {
                        "input_tokens": um.get("input_tokens", 0),
                        "output_tokens": um.get("output_tokens", 0),
                    }
        except Exception:
            pass
        return {"input_tokens": 0, "output_tokens": 0}

    def _extract_openai_usage(self, response: Any) -> dict[str, int]:
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                return {
                    "input_tokens": um.get("input_tokens", 0),
                    "output_tokens": um.get("output_tokens", 0),
                }
        except Exception:
            pass
        try:
            raw = getattr(response, "response_metadata", {}) or {}
            usage = raw.get("token_usage", {}) or raw.get("usage", {}) or {}
            if isinstance(usage, dict):
                return {
                    "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                    "output_tokens": usage.get("completion_tokens")
                    or usage.get("output_tokens")
                    or 0,
                }
        except Exception:
            pass
        return {"input_tokens": 0, "output_tokens": 0}

    def _append_jsonl(self, record: LLMCallRecord) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.session_dir / "llm_log.jsonl"
        line = record.model_dump_json() + "\n"
        with _jsonl_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)


def build_messages(system: str, user: str) -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=user)]
