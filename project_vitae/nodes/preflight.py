import logging
import os

from project_vitae.config import Config
from project_vitae.io_utils import userprofile_path
from project_vitae.latex_utils import detect_compiler, validate_template_placeholders
from project_vitae.models import ConfigError, SessionState, TemplateError

logger = logging.getLogger(__name__)


def make_preflight(cfg: Config):
    def preflight(state: SessionState) -> dict:
        for name in cfg.subagents:
            subagent = cfg.subagent(name)
            if not subagent.api_key and not (
                subagent.api_key_env and os.environ.get(subagent.api_key_env)
            ):
                source = (
                    f"environment variable '{subagent.api_key_env}'"
                    if subagent.api_key_env
                    else "API key"
                )
                raise ConfigError(f"{source} not set for subagent '{name}'")

        template_rel = cfg.latex.template_path
        template_path = userprofile_path([template_rel])
        if not template_path.is_file():
            raise TemplateError(
                f"template not found at {template_path}; "
                f"copy template.example.tex to {template_rel}"
            )

        text = template_path.read_text(encoding="utf-8")
        missing, unknown = validate_template_placeholders(text)
        if missing:
            raise TemplateError(
                f"missing required placeholders in template: {', '.join(sorted(missing))}"
            )
        if unknown:
            logger.warning("unknown placeholders in template (will pass through): %s", unknown)

        compiler = cfg.latex.compiler
        try:
            if compiler == "auto":
                compiler = detect_compiler()
            else:
                detect_compiler()
        except TemplateError:
            logger.warning("no LaTeX compiler found on PATH — PDF export will fail at compile step")
            compiler = None

        return {"latex_compiler": compiler}

    return preflight
