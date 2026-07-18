import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_template(tmp_path: Path) -> Path:
    path = tmp_path / "template.tex"
    path.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\VAR{summary}\n"
        "\\VAR{experience}\n"
        "\\VAR{education}\n"
        "\\VAR{skills}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_project_dir(tmp_path: Path) -> Path:
    proj = tmp_path / "projects" / "test-project"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "record.yaml").write_text(
        "title: Test Project\n"
        "summary: A test project\n"
        "tags:\n"
        "  - python\n"
        "  - testing\n"
        "source_repo: https://github.com/user/test\n",
        encoding="utf-8",
    )
    (proj / "summary.md").write_text("A test project\n", encoding="utf-8")
    (proj / "tags.md").write_text("python\ntesting\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def monkeypatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
