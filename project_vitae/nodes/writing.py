import logging
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage

from project_vitae.config import Config
from project_vitae.io_utils import USERPROFILE_DIR, load_project_records, read_text, slugify
from project_vitae.llm_call import LLMCall
from project_vitae.models import (
    ResumeSection,
    SectionVersion,
    SessionState,
    WritingResult,
)

logger = logging.getLogger(__name__)

SECTION_ORDER: list[Literal["experience", "education", "skills", "summary"]] = [
    "experience",
    "education",
    "skills",
    "summary",
]


def make_writing(
    cfg: Config, section_kind: Literal["experience", "education", "skills", "summary"]
):
    def writing_node(state: SessionState) -> dict:
        session_name = state.session_name or "default"
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)

        selected_titles = state.selected_projects
        records = [r for r in load_project_records() if r.title in selected_titles]

        jd = state.job_description or "(not provided)"

        try:
            userinfo_path = USERPROFILE_DIR / "userinfo.md"
            userinfo = (
                read_text(userinfo_path) if userinfo_path.is_file() else "No userinfo available."
            )
        except Exception:
            userinfo = "Error reading userinfo."

        cache = state.generated_sections_cache
        earlier_content = ""
        for k in SECTION_ORDER:
            if k == section_kind:
                break
            if k in cache:
                earlier_content += f"\n### {k}\n{cache[k]}\n"

        feedback = state.current_feedback or ""
        prev_content = ""
        for sec in state.sections:
            if sec.kind == section_kind and sec.versions:
                prev_content = sec.current.content

        user_message = (
            f"Write the '{section_kind}' section of a resume.\n\n"
            f"Job Description:\n{jd}\n\n"
            f"User Info:\n{userinfo}\n\n"
            f"Selected Projects:\n"
            + "\n".join(f"- {r.title}: {r.summary} (tags: {', '.join(r.tags)})" for r in records)
            + f"\n\nEarlier Sections:\n{earlier_content}\n"
            + (f"\nFeedback: {feedback}\n" if feedback else "")
            + (f"\nPrevious Version:\n{prev_content}\n" if prev_content else "")
            + "\nOutput WritingResult with section_id, content, and rationale."
        )

        llm_call = LLMCall(
            subagent_name="writing",
            cfg=cfg.subagent("writing"),
            session_dir=session_dir,
            output_schema=WritingResult,
            cost_guard=None,
            retry_cfg=cfg.retry,
            config=cfg,
        )

        result = llm_call.invoke([HumanMessage(content=user_message)])
        writing_result: WritingResult = result.output

        version = SectionVersion(
            content=writing_result.content,
            timestamp=datetime.now(timezone.utc),
            model=cfg.subagent("writing").model,
            provider=cfg.subagent("writing").provider,
            prompt_version=cfg.subagent("writing").prompt_version,
            temperature=cfg.subagent("writing").temperature,
            max_tokens=cfg.subagent("writing").max_tokens,
            cost_estimate=result.cost,
            feedback_used=feedback if feedback else None,
        )

        sections = list(state.sections)
        found = False
        for i, sec in enumerate(sections):
            if sec.kind == section_kind:
                sections[i].versions.append(version)
                sections[i].status = "draft"
                found = True
                break
        if not found:
            sections.append(
                ResumeSection(
                    id=f"{section_kind}_{len(sections)}",
                    kind=section_kind,
                    versions=[version],
                )
            )

        cache = dict(state.generated_sections_cache)
        cache[section_kind] = writing_result.content

        return {
            "sections": sections,
            "generated_sections_cache": cache,
            "current_section_kind": None,
            "current_feedback": None,
        }

    return writing_node
