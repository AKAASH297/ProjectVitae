from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from project_vitae.config import AppConfig
from project_vitae.latex_utils import validate_placeholders

logger = logging.getLogger(__name__)


def run_preflight(config: AppConfig, userprofile_dir: Path) -> None:
    errors: list[str] = []

    template_path = userprofile_dir / config.latex.template_path
    if not template_path.exists():
        errors.append(f"Template not found at '{template_path}'")
    else:
        template_text = template_path.read_text(encoding="utf-8")
        missing = validate_placeholders(template_text)
        if missing:
            errors.append(
                f"Template missing required placeholders: {', '.join(missing)}"
            )

    for name, sa in config.subagents.items():
        if not os.environ.get(sa.api_key_env):
            errors.append(
                f"Environment variable '{sa.api_key_env}' (subagent '{name}') not set"
            )

    compiler = _detect_latex_compiler()
    if compiler is None:
        errors.append(
            "No LaTeX compiler found. Install tectonic or pdflatex."
        )

    if errors:
        raise RuntimeError("Pre-flight checks failed:\n" + "\n".join(errors))

    logger.info("Pre-flight checks passed. LaTeX compiler: %s", compiler)


def _detect_latex_compiler() -> str | None:
    if shutil.which("tectonic"):
        return "tectonic"
    if shutil.which("pdflatex"):
        return "pdflatex"
    return None
