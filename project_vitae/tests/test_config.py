import os
from pathlib import Path

import pytest
import yaml

from project_vitae.config import AppConfig


def _write_config(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def test_minimal_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    data = {
        "subagents": {
            "explore": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_API_KEY",
                "prompt_version": "prompts/explore/v1.md",
            }
        }
    }
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, data)
    config = AppConfig.load(cfg_path)
    assert config.subagents["explore"].model == "claude-sonnet-4-6"
    assert config.retry.max_attempts == 3
    assert config.cost.per_session_cap_usd == 5.0
    assert config.latex.compiler == "auto"


def test_missing_api_key_raises(tmp_path):
    data = {
        "subagents": {
            "filter": {
                "api_key_env": "MISSING_KEY_VAR",
                "prompt_version": "prompts/filter/v1.md",
            }
        }
    }
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, data)
    with pytest.raises(ValueError, match="MISSING_KEY_VAR"):
        AppConfig.load(cfg_path)


def test_custom_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc")
    data = {
        "subagents": {
            "test": {
                "api_key_env": "TEST_KEY",
                "prompt_version": "prompts/test/v1.md",
            }
        },
        "retry": {"max_attempts": 5, "backoff_seconds": [0.5, 1, 2, 4, 8]},
        "cost": {"per_session_cap_usd": 2.0},
    }
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, data)
    config = AppConfig.load(cfg_path)
    assert config.retry.max_attempts == 5
    assert config.retry.backoff_seconds == [0.5, 1, 2, 4, 8]
    assert config.cost.per_session_cap_usd == 2.0


def test_pricing_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("KEY", "val")
    data = {
        "subagents": {
            "w": {"api_key_env": "KEY", "prompt_version": "p/v1.md"}
        },
        "cost": {
            "pricing_overrides": {
                "my-model": {"input": 1e-6, "output": 2e-6}
            }
        },
    }
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, data)
    config = AppConfig.load(cfg_path)
    assert config.cost.pricing_overrides["my-model"]["input"] == 1e-6
