import os
from pathlib import Path

import pytest
import yaml

from project_vitae.config import Config, SubagentConfig, load_config
from project_vitae.models import ConfigError


def _write_config(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _minimal_config() -> dict:
    return {
        "subagents": {
            "explore": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "claude-sonnet-4-20250514", "prompt_version": "prompts/explore/v1.md"},
            "filter": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "claude-sonnet-4-20250514", "prompt_version": "prompts/filter/v1.md"},
            "writing": {"provider": "openai_compatible", "api_key_env": "TEST_KEY", "model": "gpt-4o", "prompt_version": "prompts/writing/v1.md"},
            "content_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "claude-sonnet-4-20250514", "prompt_version": "prompts/content_critique/v1.md"},
            "compile_critique": {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "claude-sonnet-4-20250514", "prompt_version": "prompts/compile_critique/v1.md"},
        },
    }


def test_load_valid_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg_path = _write_config(tmp_path / "config.yaml", _minimal_config())
    cfg = load_config(cfg_path)
    assert len(cfg.subagents) == 5
    assert cfg.subagents["explore"].model == "claude-sonnet-4-20250514"
    assert cfg.subagents["writing"].provider == "openai_compatible"
    assert cfg.retry.max_attempts == 3
    assert cfg.cost.per_session_cap_usd == 5.00
    assert cfg.log_level == "info"


def test_load_config_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg_path = _write_config(tmp_path / "config.yaml", _minimal_config())
    cfg = load_config(cfg_path)
    assert cfg.latex.compiler == "auto"
    assert cfg.latex.template_path == "template.tex"
    assert cfg.retry.backoff_seconds == [1, 2, 4]


def test_missing_subagent_section(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    data = _minimal_config()
    del data["subagents"]["filter"]
    cfg_path = _write_config(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError, match="missing subagent"):
        load_config(cfg_path)


def test_unknown_subagent_section(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    data = _minimal_config()
    data["subagents"]["extra"] = {"provider": "anthropic", "api_key_env": "TEST_KEY", "model": "x", "prompt_version": "v1"}
    cfg_path = _write_config(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError, match="unknown subagent"):
        load_config(cfg_path)


def test_unknown_root_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    data = _minimal_config()
    data["unknown_key"] = "value"
    cfg_path = _write_config(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(cfg_path)


def test_unknown_subagent_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    data = _minimal_config()
    data["subagents"]["explore"]["unknown_field"] = "x"
    cfg_path = _write_config(tmp_path / "config.yaml", data)
    with pytest.raises(ConfigError, match="unknown keys in subagent 'explore'"):
        load_config(cfg_path)


def test_missing_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    cfg_path = _write_config(tmp_path / "config.yaml", _minimal_config())
    cfg = load_config(cfg_path)
    with pytest.raises(ConfigError, match="TEST_KEY"):
        cfg.api_key("explore")


def test_config_file_not_found(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_subagent_accessor(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")
    cfg_path = _write_config(tmp_path / "config.yaml", _minimal_config())
    cfg = load_config(cfg_path)
    sa = cfg.subagent("writing")
    assert sa.model == "gpt-4o"
    with pytest.raises(KeyError):
        cfg.subagent("nonexistent")


def test_api_key_resolved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-real-value")
    cfg_path = _write_config(tmp_path / "config.yaml", _minimal_config())
    cfg = load_config(cfg_path)
    assert cfg.api_key("explore") == "sk-real-value"
