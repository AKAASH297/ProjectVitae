from __future__ import annotations

import logging
import re
from pathlib import Path

from project_vitae.config import AppConfig
from project_vitae.io_utils import load_prompt
from project_vitae.llm_call import LLMCallError, llm_call
from project_vitae.models import CritiqueResult, Issue, SessionState

logger = logging.getLogger(__name__)

CONTENT_CRITIQUE_SYSTEM_PROMPT = """You are a resume content critique agent. Review the draft resume 
sections against the job description and identify:
1. Missing keywords or skills from the job description
2. Weak or vague statements
3. Opportunities to better align content with the job requirements

First compute keyword overlap, then assess semantic alignment."""


async def run_content_critique(
    config: AppConfig,
    session_dir: Path,
    state: SessionState,
    prompts_dir: Path,
) -> CritiqueResult:
    sub_cfg = config.subagents["content_critique"]
    prompt_path = prompts_dir / sub_cfg.prompt_version

    try:
        prompt_text = load_prompt(prompt_path)
    except FileNotFoundError:
        prompt_text = CONTENT_CRITIQUE_SYSTEM_PROMPT

    draft_content = _collect_draft_content(state)
    keyword_issues = _deterministic_keyword_overlap(state.job_description, draft_content)

    user_prompt = _build_critique_prompt(state.job_description, draft_content, keyword_issues)

    result, raw = await llm_call(
        subagent_name="content_critique",
        subagent_cfg=sub_cfg,
        app_cfg=config,
        system_prompt=prompt_text,
        user_prompt=user_prompt,
        session_dir=session_dir,
        running_cost=0.0,
        cost_cap=config.cost.per_session_cap_usd,
        output_model=CritiqueResult,
    )

    if isinstance(raw, CritiqueResult):
        combined = keyword_issues + raw.issues
        return CritiqueResult(issues=combined)
    raise LLMCallError("Content Critique subagent did not return structured output", recoverable=False)


def _deterministic_keyword_overlap(jd: str, draft: str) -> list[Issue]:
    jd_lower = jd.lower()
    draft_lower = draft.lower()
    words = set(re.findall(r"[a-z]+(?:[-/][a-z]+)*", jd_lower))
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "must", "need", "this",
        "that", "these", "those", "it", "its", "we", "you", "they", "them",
        "their", "our", "not", "no", "nor",
    }
    meaningful = words - stopwords
    found = {w for w in meaningful if w in draft_lower}
    missing = meaningful - found
    issues = []
    for word in sorted(missing):
        issues.append(
            Issue(
                location="global",
                kind="content_keyword",
                note=f"Keyword '{word}' from job description not found in draft",
                keyword_match=True,
                phase="content",
            )
        )
    return issues


def _collect_draft_content(state: SessionState) -> str:
    parts = []
    for section in state.sections:
        parts.append(f"[{section.id}]\n{section.current.content}")
    return "\n\n".join(parts)


def _build_critique_prompt(jd: str, draft: str, keyword_issues: list[Issue]) -> str:
    kw_lines = "\n".join(f"- {i.note}" for i in keyword_issues)
    return f"""Job Description:
{jd}

Current Draft Content:
{draft}

Deterministic keyword overlap issues found:
{kw_lines if keyword_issues else "None detected"}

Please also perform a semantic match assessment and identify any additional issues with:
- Weak or vague statements
- Missing alignment with job requirements
- Opportunities for improvement
"""
