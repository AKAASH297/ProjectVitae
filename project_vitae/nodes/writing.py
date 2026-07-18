from __future__ import annotations

import logging
from pathlib import Path

from project_vitae.config import AppConfig
from project_vitae.io_utils import load_prompt, read_text
from project_vitae.llm_call import LLMCallError, llm_call
from project_vitae.models import (
    ProjectRecord,
    ResumeSection,
    SectionVersion,
    SessionState,
    WritingResult,
)

logger = logging.getLogger(__name__)

WRITING_SYSTEM_PROMPT = """You are an expert resume writer. Given project information, 
user information, and a job description, write a compelling resume section. 
Be specific, use active language, and highlight achievements and impact."""

SECTION_ORDER = ["experience", "education", "skills", "summary"]


async def write_section(
    kind: str,
    config: AppConfig,
    session_dir: Path,
    state: SessionState,
    projects: list[ProjectRecord],
    userinfo_text: str,
    prompts_dir: Path,
    feedback: str | None = None,
) -> WritingResult:
    sub_cfg = config.subagents["writing"]
    prompt_path = prompts_dir / sub_cfg.prompt_version
    try:
        prompt_text = load_prompt(prompt_path)
    except FileNotFoundError:
        prompt_text = WRITING_SYSTEM_PROMPT

    user_prompt = _build_writing_prompt(
        kind, state.job_description, projects, userinfo_text, feedback
    )

    result, raw = await llm_call(
        subagent_name=f"writing_{kind}",
        subagent_cfg=sub_cfg,
        app_cfg=config,
        system_prompt=prompt_text,
        user_prompt=user_prompt,
        session_dir=session_dir,
        running_cost=0.0,
        cost_cap=config.cost.per_session_cap_usd,
        output_model=WritingResult,
    )

    if isinstance(raw, WritingResult):
        return raw
    raise LLMCallError("Writing subagent did not return structured output", recoverable=False)


def _build_writing_prompt(
    kind: str, jd: str, projects: list[ProjectRecord], userinfo: str, feedback: str | None
) -> str:
    project_lines = []
    for p in projects:
        project_lines.append(f"- {p.title}: {p.summary} (tags: {', '.join(p.tags)})")

    parts = [
        f"Section kind: {kind}",
        f"Job Description:\n{jd}",
        f"User Information:\n{userinfo}",
        f"Selected Projects:\n" + "\n".join(project_lines),
    ]
    if feedback:
        parts.append(f"User Feedback (incorporate this):\n{feedback}")

    parts.append(
        f"\nWrite the '{kind}' section of a resume. "
        "Return a section_id (e.g. 'experience', 'education'), the section content in plain text, "
        "and a short rationale for your choices."
    )
    return "\n\n".join(parts)
