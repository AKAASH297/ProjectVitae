from __future__ import annotations

import logging
from pathlib import Path

from project_vitae.config import AppConfig
from project_vitae.io_utils import load_prompt
from project_vitae.llm_call import LLMCallError, llm_call
from project_vitae.models import FilterResult, ProjectRecord, SessionState

logger = logging.getLogger(__name__)

FILTER_SYSTEM_PROMPT = """You are a resume filter agent. Given a job description and a list of 
project records, select the subset of projects that are most relevant to include in the resume 
for this job. Only include/exclude — no ranking or ordering. Explain your reasoning."""


async def run_filter(
    config: AppConfig,
    session_dir: Path,
    state: SessionState,
    projects: list[ProjectRecord],
    prompts_dir: Path,
) -> FilterResult:
    sub_cfg = config.subagents["filter"]
    prompt_path = prompts_dir / sub_cfg.prompt_version
    try:
        prompt_text = load_prompt(prompt_path)
    except FileNotFoundError:
        prompt_text = FILTER_SYSTEM_PROMPT

    user_prompt = _build_filter_prompt(state.job_description, projects)

    result, raw = await llm_call(
        subagent_name="filter",
        subagent_cfg=sub_cfg,
        app_cfg=config,
        system_prompt=prompt_text,
        user_prompt=user_prompt,
        session_dir=session_dir,
        running_cost=0.0,
        cost_cap=config.cost.per_session_cap_usd,
        output_model=FilterResult,
    )

    if isinstance(raw, FilterResult):
        return raw
    raise LLMCallError("Filter subagent did not return structured output", recoverable=False)


def _build_filter_prompt(jd: str, projects: list[ProjectRecord]) -> str:
    lines = [f"Job Description:\n{jd}\n", "Available Projects:"]
    for p in projects:
        confidence = " (low confidence)" if p.low_confidence else ""
        lines.append(
            f"\n- {p.title}{confidence}\n  Summary: {p.summary}\n  Tags: {', '.join(p.tags)}"
        )
    lines.append("\nSelect the projects most relevant to this job description.")
    return "\n".join(lines)
