from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from project_vitae.models import (
    Issue,
    ResumeSection,
    SectionVersion,
    SessionState,
)
from project_vitae.nodes.content_critique import _keyword_overlap, make_content_critique


def _make_state() -> SessionState:
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="I worked on Python projects", timestamp=now, provider="anthropic")
    section = ResumeSection(id="exp1", kind="experience", versions=[v], status="approved")
    return SessionState(
        session_name="test",
        job_description="Looking for Python and Rust developer with Kubernetes experience",
        sections=[section],
    )


def test_keyword_overlap_finds_missing():
    jd = "Python Rust Kubernetes"
    text = "I worked on Python projects"
    issues = _keyword_overlap(jd, text)
    assert len(issues) >= 2
    terms_found = {i.note for i in issues}
    assert any("rust" in n.lower() for n in terms_found)
    assert any("kubernetes" in n.lower() for n in terms_found)
    assert not any("python" in n.lower() for n in terms_found)


def test_keyword_overlap_empty_jd():
    assert _keyword_overlap("", "some text") == []


def test_keyword_overlap_all_present():
    jd = "Python Rust"
    text = "Python and Rust"
    assert _keyword_overlap(jd, text) == []


def test_content_critique_merged_issues(tmp_path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg = MagicMock()
    cfg.subagent.return_value = MagicMock(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="v1",
        temperature=0.2,
        max_tokens=4096,
        system_prompt_override=None,
    )
    cfg.retry.max_attempts = 3
    cfg.cost.pricing_overrides = {}

    from project_vitae.nodes.content_critique import LLMCall

    def fake_invoke(self, messages, prompt_override=None):
        fake_result = MagicMock()
        fake_result.output = MagicMock()
        fake_result.output.issues = [
            Issue(location="exp1", kind="content_keyword", note="tone could be more assertive", keyword_match=None, phase="content"),
        ]
        return fake_result

    with monkeypatch.context() as m:
        m.setattr("project_vitae.nodes.content_critique.LLMCall.invoke", fake_invoke)
        result = make_content_critique(cfg)(_make_state())

    assert len(result["open_issues"]) > 0
    assert any(i.kind == "content_keyword" for i in result["open_issues"])


def test_content_critique_sets_needs_review(tmp_path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg = MagicMock()
    cfg.subagent.return_value = MagicMock(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="v1",
        temperature=0.2,
        max_tokens=4096,
        system_prompt_override=None,
    )
    cfg.retry.max_attempts = 3
    cfg.cost.pricing_overrides = {}

    from project_vitae.nodes.content_critique import LLMCall

    def fake_invoke(self, messages, prompt_override=None):
        fake_result = MagicMock()
        fake_result.output = MagicMock()
        fake_result.output.issues = [
            Issue(location="global", kind="content_keyword", note="missing term", keyword_match=None, phase="content"),
        ]
        return fake_result

    state = _make_state()
    state.sections[0].status = "approved"

    with monkeypatch.context() as m:
        m.setattr("project_vitae.nodes.content_critique.LLMCall.invoke", fake_invoke)
        result = make_content_critique(cfg)(state)

    assert result["sections"][0].status == "needs_review"
