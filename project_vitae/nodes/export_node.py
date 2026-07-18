import logging

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR, atomic_write_text, read_text, slugify
from project_vitae.latex_utils import (
    compile_pdf,
    detect_compiler,
    fill_template,
    sanitize_latex,
    validate_template_placeholders,
)
from project_vitae.models import SessionState, TemplateError

logger = logging.getLogger(__name__)


def make_export(cfg: Config):
    def export(state: SessionState) -> dict:
        session_name = state.session_name or "default"
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
        session_dir.mkdir(parents=True, exist_ok=True)
        output_dir = session_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        template_rel = cfg.latex.template_path
        template_path = USERPROFILE_DIR / template_rel
        if not template_path.is_file():
            raise TemplateError(f"template not found: {template_path}")

        template = read_text(template_path)
        missing, unknown = validate_template_placeholders(template)
        if missing:
            raise TemplateError(
                f"missing required placeholders in template: {', '.join(sorted(missing))}"
            )
        if unknown:
            logger.info("unknown placeholders (will pass through): %s", unknown)

        section_content: dict[str, str] = {
            "experience": "",
            "education": "",
            "skills": "",
            "summary": "",
        }
        for sec in state.sections:
            if sec.kind in section_content and sec.versions:
                section_content[sec.kind] = sec.current.content

        for kind in section_content:
            section_content[kind] = sanitize_latex(section_content[kind])

        tex_content = fill_template(template, section_content)
        tex_path = session_dir / "resume.tex"
        atomic_write_text(tex_path, tex_content)

        compiler = state.latex_compiler or detect_compiler()
        pdf_path = compile_pdf(tex_path, output_dir, compiler)

        return {"final_pdf": str(pdf_path)}

    return export
