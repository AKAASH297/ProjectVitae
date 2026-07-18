import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR, read_text, slugify
from project_vitae.llm_call import LLMCall
from project_vitae.models import Issue, SessionState

logger = logging.getLogger(__name__)


class CompileCritiqueOutput(BaseModel):
    issues: list[Issue]


def make_compile_critique(cfg: Config):
    def compile_critique(state: SessionState) -> dict:
        session_name = state.session_name or "default"
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
        tex_path = session_dir / "resume.tex"

        if not tex_path.is_file():
            logger.warning("no resume.tex found at %s", tex_path)
            return {"open_issues": state.open_issues}

        tex_content = read_text(tex_path)
        jd = state.job_description or ""

        user_message = (
            f"Job Description:\n{jd}\n\n"
            f"Filled LaTeX Content:\n{tex_content[:10000]}\n\n"
            "Judge ATS-friendliness: check section ordering, single-column readability, "
            "no exotic packages, consistent date formats, page-break risk.\n"
            "Output issues with phase='compile'."
        )

        llm_call = LLMCall(
            subagent_name="compile_critique",
            cfg=cfg.subagent("compile_critique"),
            session_dir=session_dir,
            output_schema=CompileCritiqueOutput,
            cost_guard=None,
            retry_cfg=cfg.retry,
            config=cfg,
        )

        result = llm_call.invoke([HumanMessage(content=user_message)])
        new_issues: list[Issue] = result.output.issues

        existing = [i for i in state.open_issues if i.phase != "compile"]
        merged = existing + new_issues

        return {"open_issues": merged}

    return compile_critique
