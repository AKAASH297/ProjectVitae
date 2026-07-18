import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR, slugify
from project_vitae.llm_call import LLMCall
from project_vitae.models import CritiqueResult, Issue, SessionState

logger = logging.getLogger(__name__)


class ContentCritiqueOutput(BaseModel):
    issues: list[Issue]


def _keyword_overlap(jd: str, section_text: str) -> list[Issue]:
    jd_terms = set(
        w.lower()
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", jd or "")
    )
    if not jd_terms:
        return []
    text_lower = section_text.lower()
    missing: list[Issue] = []
    for term in sorted(jd_terms):
        if term not in text_lower:
            missing.append(Issue(
                location="global",
                kind="content_keyword",
                note=f"JD term '{term}' not found in draft content",
                keyword_match=False,
                phase="content",
            ))
    return missing


def make_content_critique(cfg: Config):
    def content_critique(state: SessionState) -> dict:
        session_name = state.session_name or "default"
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
        jd = state.job_description or ""

        all_keyword_issues: list[Issue] = []
        combined_draft = ""
        for sec in state.sections:
            if sec.versions:
                combined_draft += f"\n### {sec.kind}\n{sec.current.content}\n"

        all_keyword_issues = _keyword_overlap(jd, combined_draft)

        user_message = (
            f"Job Description:\n{jd}\n\n"
            f"Draft Resume Content:\n{combined_draft}\n\n"
            "A pre-computed keyword-coverage list is available below. "
            "Do NOT modify those entries. Add only semantic-match issues: "
            "tone, alignment, exaggeration, missing context.\n"
            "Output ContentCritiqueOutput with issues (phase=content)."
        )

        llm_call = LLMCall(
            subagent_name="content_critique",
            cfg=cfg.subagent("content_critique"),
            session_dir=session_dir,
            output_schema=ContentCritiqueOutput,
            cost_guard=None,
            retry_cfg=cfg.retry,
            config=cfg,
        )

        result = llm_call.invoke([HumanMessage(content=user_message)])
        semantic_issues: list[Issue] = result.output.issues

        merged = all_keyword_issues + semantic_issues

        sections = list(state.sections)
        for sec in sections:
            for iss in merged:
                if iss.location == sec.id or iss.location == "global":
                    if sec.status == "approved":
                        sec.status = "needs_review"

        return {
            "open_issues": merged,
            "sections": sections,
        }

    return content_critique
