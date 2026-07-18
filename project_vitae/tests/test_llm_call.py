from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from project_vitae.config import SubagentConfig, Config, RetryConfig, CostConfig, LatexConfig
from project_vitae.cost import CostGuard
from project_vitae.llm_call import LLMCall, LLMCallResult, build_messages
from project_vitae.models import CostCapReached, LLMCallError, TokenBudgetExceeded


class FakeOutput(BaseModel):
    text: str


def _make_cfg(**overrides) -> SubagentConfig:
    params = dict(
        provider="anthropic",
        api_key_env="TEST_API_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="prompts/explore/v1.md",
        temperature=0.3,
        max_tokens=4096,
    )
    params.update(overrides)
    return SubagentConfig(**params)


def _make_config() -> Config:
    return Config(
        subagents={
            "explore": _make_cfg(),
            "filter": _make_cfg(),
            "writing": _make_cfg(),
            "content_critique": _make_cfg(),
            "compile_critique": _make_cfg(),
        },
        retry=RetryConfig(),
        cost=CostConfig(),
        latex=LatexConfig(),
    )


def _make_response_with_usage(input_t: int = 100, output_t: int = 50):
    response = MagicMock(spec=FakeOutput)
    response.text = "test"
    response.response_metadata = {
        "usage": {"input_tokens": input_t, "output_tokens": output_t}
    }
    return response


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path / "session"


@pytest.fixture
def cost_guard() -> CostGuard:
    return CostGuard(100.0)


def test_happy_path(session_dir: Path, cost_guard: CostGuard, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    cfg = _make_cfg()
    llm = LLMCall("explore", cfg, session_dir, FakeOutput, cost_guard, config=_make_config())
    fake_model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = _make_response_with_usage()
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        result = llm.invoke([], prompt_override="system")
    assert isinstance(result, LLMCallResult)
    assert result.output.text == "test"
    assert cost_guard.current > 0


def test_non_recoverable_error(session_dir: Path, cost_guard: CostGuard, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    cfg = _make_cfg()
    llm = LLMCall("test", cfg, session_dir, FakeOutput, cost_guard, config=_make_config())
    fake_model = MagicMock()
    structured = MagicMock()

    class MockBadRequestError(Exception):
        pass

    structured.invoke.side_effect = MockBadRequestError("auth error")
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        with pytest.raises(LLMCallError, match="non-recoverable"):
            llm.invoke([])


def test_retry_then_succeed(session_dir: Path, cost_guard: CostGuard, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    cfg = _make_cfg()
    retry = RetryConfig(max_attempts=3, backoff_seconds=[1, 2])
    llm = LLMCall("test", cfg, session_dir, FakeOutput, cost_guard, retry_cfg=retry, config=_make_config())

    fake_model = MagicMock()
    structured = MagicMock()

    class RateLimitError(Exception):
        pass

    structured.invoke.side_effect = [
        RateLimitError("rate limited"),
        RateLimitError("rate limited"),
        _make_response_with_usage(),
    ]
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        result = llm.invoke([], prompt_override="system")
    assert result.output.text == "test"


def test_exhaust_retries(session_dir: Path, cost_guard: CostGuard, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    cfg = _make_cfg()
    retry = RetryConfig(max_attempts=2, backoff_seconds=[1])
    llm = LLMCall("test", cfg, session_dir, FakeOutput, cost_guard, retry_cfg=retry, config=_make_config())

    fake_model = MagicMock()
    structured = MagicMock()

    class RateLimitError(Exception):
        pass

    structured.invoke.side_effect = RateLimitError("always fail")
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        with pytest.raises(LLMCallError, match="exhausted retries"):
            llm.invoke([])


def test_cost_cap_raised(session_dir: Path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    guard = CostGuard(0.0001)
    cfg = _make_cfg()
    llm = LLMCall("test", cfg, session_dir, FakeOutput, guard, config=_make_config())
    fake_model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = _make_response_with_usage(1000000, 500000)
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        with pytest.raises(CostCapReached):
            llm.invoke([], prompt_override="system")


def test_token_budget(session_dir: Path, cost_guard: CostGuard, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    cfg = _make_cfg()
    accumulator = [0]
    llm = LLMCall("explore", cfg, session_dir, FakeOutput, cost_guard,
                  budget_accumulator=accumulator, budget_limit=10, config=_make_config())
    fake_model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = _make_response_with_usage(100, 50)
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        with pytest.raises(TokenBudgetExceeded):
            llm.invoke([], prompt_override="system")


def test_jsonl_appended(session_dir: Path, cost_guard: CostGuard, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    cfg = _make_cfg()
    llm = LLMCall("explore", cfg, session_dir, FakeOutput, cost_guard, config=_make_config())
    fake_model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = _make_response_with_usage()
    fake_model.with_structured_output.return_value = structured

    with patch.object(llm, "_build_model", return_value=fake_model):
        llm.invoke([], prompt_override="system")
    log_file = session_dir / "llm_log.jsonl"
    assert log_file.is_file()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) >= 1
    assert "explore" in lines[0]
    assert "claude-sonnet" in lines[0]


def test_build_messages():
    msgs = build_messages("sys", "user text")
    assert len(msgs) == 2
    assert msgs[0].content == "sys"
    assert msgs[1].content == "user text"
