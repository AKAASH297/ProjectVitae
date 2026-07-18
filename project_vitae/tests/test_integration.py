from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR
from project_vitae.latex_utils import fill_template
from project_vitae.models import (
    ExplorationResult,
    FilterResult,
    Issue,
    ResumeSection,
    SectionVersion,
    SessionState,
    WritingResult,
)

pytestmark = pytest.mark.network


def test_deterministic_keyword_overlap():
    from project_vitae.nodes.content_critique import _keyword_overlap
    jd = "Python developer with Kubernetes and Docker experience"
    text = "Worked on Python projects"
    issues = _keyword_overlap(jd, text)
    assert len(issues) >= 2
    notes = [i.note.lower() for i in issues]
    assert any("kubernetes" in n for n in notes)
    assert any("docker" in n for n in notes)
    assert not any("python" in n for n in notes)


def test_latex_sanitizer():
    from project_vitae.latex_utils import sanitize_latex
    cases = [
        ("", ""),
        ("a & b", "a \\& b"),
        ("$100", "\\$100"),
        ("{curly}", "\\{curly\\}"),
        ("_underscore", "\\_underscore"),
    ]
    for inp, expected in cases:
        assert sanitize_latex(inp) == expected


def test_placeholder_extraction():
    from project_vitae.latex_utils import extract_placeholders, validate_template_placeholders

    tpl = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}"
    assert extract_placeholders(tpl) == {"experience", "education", "skills", "summary"}

    missing, unknown = validate_template_placeholders(tpl)
    assert missing == set()
    assert unknown == set()


def test_placeholder_validation_missing():
    from project_vitae.latex_utils import validate_template_placeholders
    missing, unknown = validate_template_placeholders(r"\VAR{experience}")
    assert "education" in missing
    assert "summary" in missing


def test_fill_template_passthrough():
    tpl = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}\VAR{custom}"
    result = fill_template(tpl, {"experience": "Worked", "education": "Edu", "skills": "Skills", "summary": "Sum"})
    assert "Worked" in result
    assert r"\VAR{custom}" in result


def test_session_state_round_trip():
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="test", timestamp=now, provider="anthropic", cost_estimate=0.01)
    section = ResumeSection(id="exp1", kind="experience", versions=[v], status="approved")
    state = SessionState(
        session_name="test",
        job_description="JD text",
        selected_projects=["proj1"],
        sections=[section],
        github_urls=["https://github.com/user/repo"],
    )
    d = state.model_dump()
    state2 = SessionState.model_validate(d)
    assert state2.session_name == "test"
    assert state2.sections[0].current.cost_estimate == 0.01


def test_config_yaml_format(tmp_path):
    from project_vitae.config import load_config
    import yaml
    import os

    cfg_data = {
        "subagents": {
            "explore": {"provider": "anthropic", "api_key_env": "TEST_API_KEY", "model": "c", "prompt_version": "v1"},
            "filter": {"provider": "anthropic", "api_key_env": "TEST_API_KEY", "model": "c", "prompt_version": "v1"},
            "writing": {"provider": "anthropic", "api_key_env": "TEST_API_KEY", "model": "c", "prompt_version": "v1"},
            "content_critique": {"provider": "anthropic", "api_key_env": "TEST_API_KEY", "model": "c", "prompt_version": "v1"},
            "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_API_KEY", "model": "c", "prompt_version": "v1"},
        },
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg_data, f)

    os.environ["TEST_API_KEY"] = "xxx"

    try:
        cfg = load_config(p)
        assert len(cfg.subagents) == 5
    finally:
        del os.environ["TEST_API_KEY"]
