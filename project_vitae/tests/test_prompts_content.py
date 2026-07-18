from pathlib import Path

import pytest

from project_vitae.prompts import load_prompt


PROMPT_FILES = [
    ("prompts/explore/v1.md", "ExplorationResult"),
    ("prompts/filter/v1.md", "FilterResult"),
    ("prompts/writing/v1.md", "WritingResult"),
    ("prompts/content_critique/v1.md", "Output"),
    ("prompts/compile_critique/v1.md", "CritiqueResult"),
]


@pytest.mark.parametrize("path,expected", PROMPT_FILES)
def test_prompt_exists_and_has_content(path: str, expected: str, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("project_vitae.io_utils.USERPROFILE_DIR", tmp_path)
    p = tmp_path / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {path}\nOutput: {expected}")
    result = load_prompt(path)
    assert result
    assert "TODO" not in result
