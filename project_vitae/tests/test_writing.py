from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from project_vitae.models import ResumeSection, SectionVersion, SessionState, WritingResult
from project_vitae.nodes.writing import make_writing


def _make_state(section_kind="experience") -> SessionState:
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="Old content", timestamp=now, provider="anthropic")
    section = ResumeSection(id="exp1", kind=section_kind, versions=[v])
    return SessionState(
        session_name="test",
        job_description="Python developer",
        selected_projects=["proj1"],
        sections=[section] if section_kind == "experience" else [],
        generated_sections_cache={},
        current_feedback=None,
    )


def test_writing_creates_section(tmp_path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg = MagicMock()
    cfg.subagent.return_value = MagicMock(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="v1",
        temperature=0.5,
        max_tokens=8192,
        system_prompt_override=None,
    )
    cfg.retry.max_attempts = 3
    cfg.cost.pricing_overrides = {}

    from project_vitae.nodes.writing import LLMCall

    def fake_invoke(self, messages, prompt_override=None):
        fake_result = MagicMock()
        fake_result.output = WritingResult(
            section_id="skills1",
            content="Python, JavaScript, Go",
            rationale="relevant to JD",
        )
        fake_result.cost = 0.05
        return fake_result

    with monkeypatch.context() as m:
        m.setattr("project_vitae.nodes.writing.LLMCall.invoke", fake_invoke)
        result = make_writing(cfg, "skills")(SessionState(
            session_name="test",
            job_description="Python developer",
            selected_projects=[],
            generated_sections_cache={},
        ))

    assert len(result["sections"]) == 1
    assert result["sections"][0].kind == "skills"
    assert result["sections"][0].current.content == "Python, JavaScript, Go"


def test_writing_appends_version(tmp_path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg = MagicMock()
    cfg.subagent.return_value = MagicMock(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="v1",
        temperature=0.5,
        max_tokens=8192,
        system_prompt_override=None,
    )
    cfg.retry.max_attempts = 3
    cfg.cost.pricing_overrides = {}
    state = _make_state("experience")

    from project_vitae.nodes.writing import LLMCall

    def fake_invoke(self, messages, prompt_override=None):
        fake_result = MagicMock()
        fake_result.output = WritingResult(
            section_id="exp1",
            content="New content",
            rationale="updated",
        )
        fake_result.cost = 0.03
        return fake_result

    with monkeypatch.context() as m:
        m.setattr("project_vitae.nodes.writing.LLMCall.invoke", fake_invoke)
        result = make_writing(cfg, "experience")(state)

    assert len(result["sections"]) == 1
    assert len(result["sections"][0].versions) == 2
    assert result["sections"][0].current.content == "New content"
    assert result["sections"][0].status == "draft"
