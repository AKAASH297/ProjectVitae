from pathlib import Path

import pytest

from project_vitae.config import SubagentConfig
from project_vitae.models import PromptError
from project_vitae.prompts import ensure_prompt_path_is_safe, load_prompt, resolve_prompt


def test_ensure_safe_valid():
    ensure_prompt_path_is_safe("prompts/explore/v1.md")


def test_ensure_safe_traversal():
    with pytest.raises(PromptError, match="path traversal"):
        ensure_prompt_path_is_safe("prompts/../../etc/passwd")


def test_ensure_safe_absolute():
    with pytest.raises(PromptError, match="absolute path"):
        ensure_prompt_path_is_safe("/etc/passwd")


def test_load_prompt_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    monkeypatch.setattr("project_vitae.io_utils.get_userprofile_dir", lambda: tmp_path)
    with pytest.raises(PromptError, match="not found"):
        load_prompt("nonexistent/v1.md")


def test_load_prompt_existing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.get_userprofile_dir", lambda: tmp_path)
    prompt_dir = tmp_path / "prompts" / "explore"
    prompt_dir.mkdir(parents=True)
    prompt_file = prompt_dir / "v1.md"
    prompt_file.write_text("you are an explore agent")
    result = load_prompt("prompts/explore/v1.md")
    assert result == "you are an explore agent"


def test_resolve_prompt_override():
    cfg = SubagentConfig(
        provider="anthropic",
        api_key_env="KEY",
        model="claude-3",
        prompt_version="prompts/explore/v1.md",
        system_prompt_override="custom system prompt",
    )
    result = resolve_prompt("explore", cfg)
    assert result == "custom system prompt"


def test_resolve_prompt_from_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.get_userprofile_dir", lambda: tmp_path)
    prompt_dir = tmp_path / "prompts" / "filter"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "v1.md").write_text("filter prompt content")
    cfg = SubagentConfig(
        provider="anthropic",
        api_key_env="KEY",
        model="claude-3",
        prompt_version="prompts/filter/v1.md",
    )
    result = resolve_prompt("filter", cfg)
    assert result == "filter prompt content"
