import logging

from langchain_core.messages import HumanMessage

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR, load_project_records, slugify
from project_vitae.llm_call import LLMCall
from project_vitae.models import FilterResult, SessionState

logger = logging.getLogger(__name__)


def make_filter(cfg: Config):
    def filter_node(state: SessionState) -> dict:
        records = load_project_records()
        if not records:
            logger.warning("no project records found — filter will return empty selection")

        projects_summary = "\n".join(
            f"- {r.title}: {r.summary[:200]} (tags: {', '.join(r.tags[:10])})" for r in records
        )
        jd = state.job_description or "(no job description provided)"

        user_message = (
            f"Job Description:\n{jd}\n\n"
            f"Available Projects:\n{projects_summary}\n\n"
            "Select the projects most relevant to this job description. "
            "Output FilterResult with selected titles and rationale."
        )

        session_name = state.session_name or "default"
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
        cost_guard = None

        llm_call = LLMCall(
            subagent_name="filter",
            cfg=cfg.subagent("filter"),
            session_dir=session_dir,
            output_schema=FilterResult,
            cost_guard=cost_guard,
            retry_cfg=cfg.retry,
            config=cfg,
        )

        result = llm_call.invoke([HumanMessage(content=user_message)])
        filter_result = result.output

        return {"filter_proposal": filter_result}

    return filter_node
