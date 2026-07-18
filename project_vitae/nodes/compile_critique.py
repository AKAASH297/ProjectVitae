from __future__ import annotations

import logging
from pathlib import Path

from project_vitae.config import AppConfig
from project_vitae.io_utils import load_prompt
from project_vitae.llm_call import LLMCallError, llm_call
from project_vitae.models import CritiqueResult, SessionState

logger = logging.getLogger(__name__)

COMPILE_CRITIQUE_SYSTEM_PROMPT = """You are a resume formatting critique agent. Review the rendered 
LaTeX source of a resume and assess:
1. ATS-friendliness (can a parser extract the text correctly?)
2. Formatting issues that might affect readability
3. Layout problems

Focus on the LaTeX markup, not the content itself."""


async def run_compile_critique(
    config: AppConfig,
    session_dir: Path,
    state: SessionState,
    tex_path: Path,
    prompts_dir: Path,
) -> CritiqueResult:
    sub_cfg = config.subagents["compile_critique"]
    prompt_path = prompts_dir / sub_cfg.prompt_version

    try:
        prompt_text = load_prompt(prompt_path)
    except FileNotFoundError:
        prompt_text = COMPILE_CRITIQUE_SYSTEM_PROMPT

    tex_content = tex_path.read_text(encoding="utf-8")

    user_prompt = _build_compile_prompt(state.job_description, tex_content)

    result, raw = await llm_call(
        subagent_name="compile_critique",
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
        return raw
    raise LLMCallError("Compile Critique subagent did not return structured output", recoverable=False)


def _build_compile_prompt(jd: str, tex_content: str) -> str:
    return f"""Job Description:
{jd}

Rendered LaTeX source:
{tex_content}

Review the LaTeX source for formatting issues that might affect ATS-friendliness or readability.
Focus on:
- Proper section structure
- Clean formatting that ATS parsers can handle
- Any LaTeX constructs that might cause rendering problems
"""
