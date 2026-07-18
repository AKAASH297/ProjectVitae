from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from project_vitae.config import Config, LatexConfig, RetryConfig, CostConfig
from project_vitae.models import ResumeSection, SectionVersion, SessionState, TemplateError
from project_vitae.nodes import export_node
from project_vitae.nodes.export_node import make_export


def _make_state() -> SessionState:
    now = datetime.now(timezone.utc)
    v = SectionVersion(content="Worked on $100M projects & more", timestamp=now, provider="anthropic")
    section = ResumeSection(id="exp1", kind="experience", versions=[v], status="approved")
    edu = ResumeSection(id="edu1", kind="education", versions=[SectionVersion(content="MIT", timestamp=now, provider="manual")], status="approved")
    skills = ResumeSection(id="skill1", kind="skills", versions=[SectionVersion(content="Python, Go", timestamp=now, provider="manual")], status="approved")
    summary = ResumeSection(id="sum1", kind="summary", versions=[SectionVersion(content="Senior engineer", timestamp=now, provider="anthropic")], status="approved")
    return SessionState(
        session_name="test",
        sections=[section, edu, skills, summary],
        latex_compiler=None,
    )


def _make_cfg() -> MagicMock:
    cfg = MagicMock(spec=Config)
    cfg.latex = LatexConfig(template_path="template.tex")
    cfg.retry = RetryConfig()
    cfg.cost = CostConfig()
    return cfg


def test_export_fills_and_compiles(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(export_node, "USERPROFILE_DIR", tmp_path)
    tpl = tmp_path / "template.tex"
    tpl.write_text(
        r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}"
    )
    cfg = _make_cfg()

    with patch("project_vitae.nodes.export_node.detect_compiler", return_value="pdflatex"), \
         patch("project_vitae.nodes.export_node.compile_pdf") as mock_compile:
        mock_compile.return_value = tmp_path / "sessions" / "test" / "output" / "resume.pdf"
        result = make_export(cfg)(_make_state())

    tex_path = tmp_path / "sessions" / "test" / "resume.tex"
    assert tex_path.is_file()
    tex_content = tex_path.read_text()
    assert "\\$100M" in tex_content
    assert "\\&" in tex_content
    assert "MIT" in tex_content
    assert "Python, Go" in tex_content
    assert "Senior engineer" in tex_content
    assert "final_pdf" in result


def test_export_missing_template_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(export_node, "USERPROFILE_DIR", tmp_path)
    cfg = _make_cfg()
    cfg.latex.template_path = "nonexistent.tex"

    with pytest.raises(TemplateError, match="not found"):
        make_export(cfg)(_make_state())


def test_export_missing_placeholder_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(export_node, "USERPROFILE_DIR", tmp_path)
    tpl = tmp_path / "template.tex"
    tpl.write_text(r"\VAR{experience}")
    cfg = _make_cfg()

    with pytest.raises(TemplateError, match="missing required"):
        make_export(cfg)(_make_state())
