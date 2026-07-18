from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from project_vitae.config import AppConfig, SubagentConfig
from project_vitae.cost import estimate_cost

logger = logging.getLogger(__name__)


def _build_model(config: SubagentConfig) -> BaseChatModel:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.provider == "anthropic":
        return ChatAnthropic(**kwargs)
    return ChatOpenAI(**kwargs)


RECOVERABLE = {429, 500, 502, 503, 504}


class LLMResult(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_ms: int
    prompt_version: str
    model: str


class LLMCallError(Exception):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


async def llm_call(
    subagent_name: str,
    subagent_cfg: SubagentConfig,
    app_cfg: AppConfig,
    system_prompt: str,
    user_prompt: str,
    session_dir: Path,
    running_cost: float,
    cost_cap: float,
    output_model: type[BaseModel] | None = None,
) -> tuple[LLMResult, BaseModel | str]:
    if running_cost >= cost_cap:
        raise LLMCallError(
            f"Cost cap ${cost_cap:.2f} exceeded (running: ${running_cost:.2f})",
            recoverable=False,
        )

    model = _build_model(subagent_cfg)
    prompt_version = subagent_cfg.prompt_version

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    started = time.monotonic()
    input_tokens_total = 0
    output_tokens_total = 0

    for attempt in range(1, app_cfg.retry.max_attempts + 1):
        try:
            if output_model:
                structured = model.with_structured_output(output_model)
                result = await structured.ainvoke(messages)
                content = result.model_dump_json()
                raw = result
            else:
                result = await model.ainvoke(messages)
                content = result.content if hasattr(result, "content") else str(result)
                raw = content

            usage = {}
            if hasattr(result, "usage_metadata") and result.usage_metadata:
                usage = result.usage_metadata
            elif hasattr(result, "response_metadata") and result.response_metadata:
                usage = result.response_metadata

            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0

            duration = int((time.monotonic() - started) * 1000)

            usd_cost = estimate_cost(
                subagent_cfg.model,
                input_tokens,
                output_tokens,
                app_cfg.cost.pricing_overrides,
            )

            llm_result = LLMResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=usd_cost,
                duration_ms=duration,
                prompt_version=prompt_version,
                model=subagent_cfg.model,
            )

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "subagent": subagent_name,
                "model": subagent_cfg.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": usd_cost,
                "duration_ms": duration,
                "prompt_version": prompt_version,
            }
            _append_llm_log(session_dir, log_entry)

            return llm_result, raw

        except Exception as exc:
            is_recoverable = _is_recoverable(exc)
            if not is_recoverable or attempt == app_cfg.retry.max_attempts:
                raise LLMCallError(str(exc), recoverable=is_recoverable) from exc
            backoff = app_cfg.retry.backoff_seconds[min(attempt - 1, len(app_cfg.retry.backoff_seconds) - 1)]
            logger.warning(
                "LLM call %s attempt %d failed: %s — retrying in %.1fs",
                subagent_name, attempt, exc, backoff,
            )
            time.sleep(backoff)


def _is_recoverable(exc: Exception) -> bool:
    msg = str(exc).lower()
    for code in RECOVERABLE:
        if str(code) in msg:
            return True
    if "timeout" in msg or "rate limit" in msg:
        return True
    if "429" in msg:
        return True
    if "5" in msg[:10] and any(c.isdigit() for c in msg[:10]):
        for code in {500, 502, 503, 504}:
            if str(code) in msg:
                return True
    return False


def _append_llm_log(session_dir: Path, entry: dict) -> None:
    log_path = session_dir / "llm_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
