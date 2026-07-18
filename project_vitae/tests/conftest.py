import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

# Ensure USERPROFILE_DIR resolves to a sensible default during tests
# so modules that import it at load time get a valid path.
os.environ.setdefault("PROJECTVITAE_USERPROFILE", str(Path.cwd() / "userprofile"))

from project_vitae.config import Config, SubagentConfig


@pytest.fixture
def tmp_userprofile(tmp_path: Path) -> Path:
    up = tmp_path / "userprofile"
    up.mkdir()
    (up / "projects").mkdir()
    (up / "sessions").mkdir()
    (up / "clones").mkdir()
    prompts_dir = up / "prompts"
    for name in ("explore", "filter", "writing", "content_critique", "compile_critique"):
        (prompts_dir / name).mkdir(parents=True)
        (prompts_dir / name / "v1.md").write_text(f"# {name} prompt v1\n")
    return up


@pytest.fixture
def sample_config() -> dict[str, Any]:
    return {
        "subagents": {
            "explore": {
                "provider": "anthropic",
                "base_url": None,
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-20250514",
                "prompt_version": "prompts/explore/v1.md",
                "temperature": 0.3,
                "max_tokens": 4096,
                "per_repo_token_budget": 32000,
            },
            "filter": {
                "provider": "anthropic",
                "base_url": None,
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-20250514",
                "prompt_version": "prompts/filter/v1.md",
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            "writing": {
                "provider": "anthropic",
                "base_url": None,
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-20250514",
                "prompt_version": "prompts/writing/v1.md",
                "temperature": 0.5,
                "max_tokens": 8192,
            },
            "content_critique": {
                "provider": "anthropic",
                "base_url": None,
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-20250514",
                "prompt_version": "prompts/content_critique/v1.md",
                "temperature": 0.2,
                "max_tokens": 4096,
            },
            "compile_critique": {
                "provider": "anthropic",
                "base_url": None,
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-20250514",
                "prompt_version": "prompts/compile_critique/v1.md",
                "temperature": 0.2,
                "max_tokens": 4096,
            },
        },
        "retry": {"max_attempts": 3, "backoff_seconds": [1, 2, 4]},
        "cost": {"per_session_cap_usd": 5.00},
        "latex": {"template_path": "template.tex", "compiler": "auto"},
        "log_level": "info",
    }


@pytest.fixture
def mock_chat_model() -> MagicMock:
    model = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = MagicMock()
    model.with_structured_output.return_value = structured
    return model


@pytest.fixture
def patch_userprofile(monkeypatch: pytest.MonkeyPatch, tmp_userprofile: Path) -> None:
    monkeypatch.setenv("PROJECTVITAE_USERPROFILE", str(tmp_userprofile))
