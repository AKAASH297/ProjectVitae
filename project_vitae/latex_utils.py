import logging
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, TemplateNotFound, Undefined

from project_vitae.models import TemplateError

logger = logging.getLogger(__name__)

LATEX_SPECIALS: dict[str, str] = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}

LATEX_SPECIAL_RE = re.compile("|".join(re.escape(c) for c in LATEX_SPECIALS))

PLACEHOLDER_RE = re.compile(r"\\VAR\{(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\}")

REQUIRED_PLACEHOLDERS: frozenset[str] = frozenset({"experience", "education", "skills", "summary"})


def _escape_match(m: re.Match[str]) -> str:
    return LATEX_SPECIALS[m.group(0)]


def sanitize_latex(text: str) -> str:
    return LATEX_SPECIAL_RE.sub(_escape_match, text)


def extract_placeholders(template: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(template))


def validate_template_placeholders(template: str) -> tuple[set[str], set[str]]:
    found = extract_placeholders(template)
    missing = REQUIRED_PLACEHOLDERS - found
    unknown = found - REQUIRED_PLACEHOLDERS
    return missing, unknown


class _PassthroughUndefined(Undefined):
    def __str__(self) -> str:
        return f"\\VAR{{{self._undefined_name}}}"


def fill_template(template: str, sections: dict[str, str]) -> str:
    existing = extract_placeholders(template)
    required = REQUIRED_PLACEHOLDERS & existing
    missing_required = REQUIRED_PLACEHOLDERS - set(sections.keys())
    if missing_required:
        raise TemplateError(f"missing required placeholders: {', '.join(sorted(missing_required))}")

    env = Environment(
        variable_start_string="\\VAR{",
        variable_end_string="}",
        undefined=_PassthroughUndefined,
    )

    t = env.from_string(template)

    context = {}
    for name in existing:
        if name in sections:
            context[name] = sections[name]

    return t.render(**context)


@lru_cache(maxsize=1)
def detect_compiler() -> str:
    if shutil.which("tectonic"):
        logger.info("detected LaTeX compiler: tectonic")
        return "tectonic"
    if shutil.which("pdflatex"):
        logger.info("detected LaTeX compiler: pdflatex")
        return "pdflatex"
    raise TemplateError("no LaTeX compiler found (install tectonic or pdflatex)")


def compile_pdf(tex_path: Path, out_dir: Path, compiler: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_path.resolve()
    out_dir = out_dir.resolve()

    if compiler == "tectonic":
        result = subprocess.run(
            ["tectonic", "-o", str(out_dir), str(tex_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    elif compiler == "pdflatex":
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(out_dir), str(tex_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
    else:
        raise TemplateError(f"unknown compiler: {compiler}")

    if result.returncode != 0:
        log = result.stdout[-2000:] + result.stderr[-2000:]
        lines = log.splitlines()[-30:]
        raise TemplateError(f"LaTeX compilation failed with {compiler}\n" + "\n".join(lines))

    pdf_name = tex_path.stem + ".pdf"
    pdf_path = out_dir / pdf_name
    if not pdf_path.is_file():
        raise TemplateError(f"PDF not produced at {pdf_path}")
    return pdf_path
