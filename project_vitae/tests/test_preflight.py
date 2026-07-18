from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from project_vitae.config import Config, load_config
from project_vitae.io_utils import get_userprofile_dir
from project_vitae.models import ConfigError, SessionState, TemplateError
from project_vitae.nodes.preflight import make_preflight
from project_vitae.nodes import preflight as preflight_module


def _make_state() -> SessionState:
    return SessionState(session_name="test-session")


def _minimal_cfg_data() -> dict:
    return {
        "subagents": {
            "explore": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "filter": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "writing": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "content_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
        },
        "latex": {"template_path": "template.tex", "compiler": "auto"},
    }


def test_preflight_passes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    monkeypatch.setattr("project_vitae.nodes.preflight.userprofile_path", lambda parts: tmp_path / parts[0])

    tpl = tmp_path / "template.tex"
    tpl.write_text(r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}")

    with patch("project_vitae.nodes.preflight.detect_compiler", return_value="pdflatex"):
        from project_vitae.config import Config
        cfg = Config(
            subagents={
                "explore": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
                "filter": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
                "writing": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
                "content_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
                "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            },
        )
        cfg.latex.template_path = "template.tex"
        cfg.latex.compiler = "auto"
        result = make_preflight(cfg)(_make_state())
    assert result["latex_compiler"] == "pdflatex"


def test_preflight_missing_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.setattr("project_vitae.nodes.preflight.userprofile_path", lambda parts: tmp_path / parts[0])

    tpl = tmp_path / "template.tex"
    tpl.write_text(r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}")

    from project_vitae.config import Config
    cfg = Config(
        subagents={
            "explore": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "filter": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "writing": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "content_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
        },
    )

    with pytest.raises(ConfigError, match="TEST_KEY"):
        make_preflight(cfg)(_make_state())


def test_preflight_missing_template(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    monkeypatch.setattr("project_vitae.nodes.preflight.userprofile_path", lambda parts: tmp_path / parts[0])

    from project_vitae.config import Config
    cfg = Config(
        subagents={
            "explore": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "filter": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "writing": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "content_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
        },
    )
    cfg.latex.template_path = "template.tex"

    with pytest.raises(TemplateError, match="not found"):
        make_preflight(cfg)(_make_state())


def test_preflight_missing_placeholder(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    monkeypatch.setattr("project_vitae.nodes.preflight.userprofile_path", lambda parts: tmp_path / parts[0])

    tpl = tmp_path / "template.tex"
    tpl.write_text(r"\VAR{experience}")

    from project_vitae.config import Config
    cfg = Config(
        subagents={
            "explore": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "filter": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "writing": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "content_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
            "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "c", "prompt_version": "v1"},
        },
    )
    cfg.latex.template_path = "template.tex"

    with pytest.raises(TemplateError, match="missing required"):
        make_preflight(cfg)(_make_state())
