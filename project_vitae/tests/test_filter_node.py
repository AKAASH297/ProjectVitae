from unittest.mock import MagicMock

from project_vitae.models import FilterResult, SessionState
from project_vitae.nodes.filter_node import make_filter


def _make_state() -> SessionState:
    return SessionState(session_name="test", job_description="Looking for Python developer")


def test_filter_node_returns_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg = MagicMock()
    cfg.subagent.return_value = MagicMock(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="prompts/filter/v1.md",
        temperature=0.1,
        max_tokens=2048,
    )
    cfg.retry.max_attempts = 3
    cfg.cost.pricing_overrides = {}
    cfg.subagent.return_value.system_prompt_override = None

    def fake_invoke(self, messages, prompt_override=None):
        fake_result = MagicMock()
        fake_result.output = FilterResult(selected=["proj1"], rationale="best match")
        return fake_result

    with monkeypatch.context() as m:
        m.setattr("project_vitae.nodes.filter_node.LLMCall.invoke", fake_invoke)
        result = make_filter(cfg)(_make_state())

    assert "filter_proposal" in result
    assert result["filter_proposal"].selected == ["proj1"]


def test_filter_empty_projects(tmp_path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg = MagicMock()
    cfg.subagent.return_value = MagicMock(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="v1",
        temperature=0.1,
        max_tokens=2048,
        system_prompt_override=None,
    )
    cfg.retry.max_attempts = 3
    cfg.cost.pricing_overrides = {}

    def fake_invoke(self, messages, prompt_override=None):
        fake_result = MagicMock()
        fake_result.output = FilterResult(selected=[], rationale="no relevant projects")
        return fake_result

    with monkeypatch.context() as m:
        m.setattr("project_vitae.nodes.filter_node.LLMCall.invoke", fake_invoke)
        result = make_filter(cfg)(_make_state())

    assert result["filter_proposal"].selected == []
