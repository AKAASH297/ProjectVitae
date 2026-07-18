from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from project_vitae.io_utils import read_text
from project_vitae.latex_utils import fill_template, sanitize_latex, validate_placeholders
from project_vitae.models import SessionState

logger = logging.getLogger(__name__)


def run_export(
    state: SessionState,
    template_path: Path,
    session_dir: Path,
    compiler: str,
) -> Path:
    template = read_text(template_path)

    missing = validate_placeholders(template)
    if missing:
        raise ValueError(
            f"Template missing required placeholders: {', '.join(missing)}"
        )

    sections = {}
    for section in state.sections:
        sanitized = sanitize_latex(section.current.content)
        sections[section.id] = sanitized

    filled = fill_template(template, sections)

    tex_path = session_dir / "resume.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(filled, encoding="utf-8")

    output_dir = session_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if compiler == "tectonic":
        _run_tectonic(tex_path, output_dir)
    elif compiler == "pdflatex":
        _run_pdflatex(tex_path, output_dir)
    else:
        raise ValueError(f"Unknown compiler: {compiler}")

    pdf_path = output_dir / "resume.pdf"
    if not pdf_path.exists():
        raise RuntimeError("Compilation failed — PDF not produced")

    return pdf_path


def _run_tectonic(tex_path: Path, output_dir: Path) -> None:
    result = subprocess.run(
        ["tectonic", "-o", str(output_dir), str(tex_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        log_content = result.stdout[-2000:] if result.stdout else result.stderr[-2000:]
        raise RuntimeError(f"tectonic failed:\n{log_content}")


def _run_pdflatex(tex_path: Path, output_dir: Path) -> None:
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            capture_output=True, text=True, timeout=120,
        )
    if result.returncode != 0:
        log_content = result.stdout[-2000:] if result.stdout else result.stderr[-2000:]
        raise RuntimeError(f"pdflatex failed:\n{log_content}")
