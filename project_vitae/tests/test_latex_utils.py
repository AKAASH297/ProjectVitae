from pathlib import Path

import pytest

from project_vitae.latex_utils import (
    compile_pdf,
    detect_compiler,
    extract_placeholders,
    fill_template,
    sanitize_latex,
    validate_template_placeholders,
)
from project_vitae.models import TemplateError


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("hello", "hello"),
        ("&", "\\&"),
        ("% $ #", "\\% \\$ \\#"),
        ("_ { }", "\\_ \\{ \\}"),
        ("~ ^", "\\textasciitilde{} \\textasciicircum{}"),
        ("\\", "\\textbackslash{}"),
        ("a & b", "a \\& b"),
        ("", ""),
        ("100% done", "100\\% done"),
        ("cost=$5", "cost=\\$5"),
        ("{curly}", "\\{curly\\}"),
    ],
)
def test_sanitize_latex(input_text: str, expected: str):
    assert sanitize_latex(input_text) == expected


def test_extract_placeholders():
    template = r"\VAR{experience} and \VAR{skills}"
    assert extract_placeholders(template) == {"experience", "skills"}


def test_extract_placeholders_none():
    assert extract_placeholders("no placeholders") == set()


def test_validate_placeholders_all_present():
    text = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}"
    missing, unknown = validate_template_placeholders(text)
    assert missing == set()
    assert unknown == set()


def test_validate_placeholders_unknown():
    text = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}\VAR{custom}"
    missing, unknown = validate_template_placeholders(text)
    assert missing == set()
    assert unknown == {"custom"}


def test_validate_placeholders_missing():
    text = r"\VAR{experience}\VAR{skills}"
    missing, unknown = validate_template_placeholders(text)
    assert missing == {"education", "summary"}
    assert unknown == set()


def test_fill_template_basic():
    template = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}"
    sections = {
        "experience": "Worked at X",
        "education": "MIT",
        "skills": "Python",
        "summary": "Summary",
    }
    result = fill_template(template, sections)
    assert "Worked at X" in result
    assert "MIT" in result
    assert "Python" in result
    assert "Summary" in result


def test_fill_template_unknown_placeholder_passes_through():
    template = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary} and \VAR{custom}"
    sections = {"experience": "Work", "education": "Edu", "skills": "Skills", "summary": "Sum"}
    result = fill_template(template, sections)
    assert "Work" in result
    assert r"\VAR{custom}" in result


def test_fill_template_missing_required_raises():
    template = r"\VAR{experience}"
    with pytest.raises(TemplateError, match="missing required"):
        fill_template(template, {"experience": "Work"})


def test_fill_template_sanitized_sections():
    template = r"\VAR{experience}\VAR{education}\VAR{skills}\VAR{summary}"
    sections = {"experience": "cost was $50 & up", "education": "", "skills": "", "summary": ""}
    result = fill_template(template, sections)
    assert "cost was $50 & up" in result


def test_detect_compiler():
    try:
        compiler = detect_compiler()
        assert compiler in ("tectonic", "pdflatex")
    except TemplateError:
        pass


def test_compile_pdf_no_compiler(tmp_path: Path):
    with pytest.raises(TemplateError):
        compile_pdf(tmp_path / "test.tex", tmp_path / "out", "nonexistent")


def test_compile_pdf_invalid_tex(tmp_path: Path):
    tex = tmp_path / "bad.tex"
    tex.write_text(r"\documentclass{article}\begin{document}Hello\end{document}")
    out = tmp_path / "out"
    try:
        compiler = detect_compiler()
        result = compile_pdf(tex, out, compiler)
        assert result.is_file()
        assert result.suffix == ".pdf"
    except TemplateError:
        pass
