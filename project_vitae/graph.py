from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from project_vitae.config import AppConfig
from project_vitae.models import (
    CritiqueResult,
    ExplorationResult,
    FilterResult,
    ProjectRecord,
    ResumeSection,
    SectionVersion,
    SessionState,
    WritingResult,
)
from project_vitae.nodes.preflight import run_preflight, _detect_latex_compiler
from project_vitae.nodes.clone import clone_repo
from project_vitae.nodes.explore import explore_repo
from project_vitae.nodes.filter_node import run_filter
from project_vitae.nodes.writing import write_section, SECTION_ORDER
from project_vitae.nodes.content_critique import run_content_critique
from project_vitae.nodes.compile_critique import run_compile_critique
from project_vitae.nodes.export_node import run_export

logger = logging.getLogger(__name__)


class GraphState(BaseModel):
    userprofile_dir: Path = Path("userprofile")
    session_dir: Path = Path("userprofile/sessions/default")
    config: AppConfig | None = None

    session: SessionState = Field(default_factory=SessionState)

    repo_urls: list[str] = Field(default_factory=list)
    repo_index: int = 0
    cloned_paths: dict[str, Path] = Field(default_factory=dict)
    exploration_results: list[ExplorationResult] = Field(default_factory=list)

    filter_result: FilterResult | None = None
    writing_results: dict[str, WritingResult] = Field(default_factory=dict)
    content_critique_result: CritiqueResult | None = None
    compile_critique_result: CritiqueResult | None = None
    export_pdf: str | None = None

    user_interrupt: str | None = None
    feedback: str | None = None
    skip_repo: bool = False
    abort_session: bool = False


def _load_projects(state: GraphState) -> list[ProjectRecord]:
    projects_dir = state.userprofile_dir / "projects"
    records: list[ProjectRecord] = []
    if not projects_dir.exists():
        return records
    for proj_dir in projects_dir.iterdir():
        record_path = proj_dir / "record.yaml"
        if record_path.exists():
            try:
                import yaml
                with open(record_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                records.append(ProjectRecord.model_validate(data))
            except Exception as e:
                logger.warning("Failed to load project record %s: %s", record_path, e)
    return records


async def preflight_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    run_preflight(state.config, state.userprofile_dir)
    compiler = _detect_latex_compiler()
    return {"user_interrupt": None}


async def clone_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    if state.repo_index >= len(state.repo_urls):
        return {"repo_index": state.repo_index + 1}

    url = state.repo_urls[state.repo_index]
    clones_dir = state.userprofile_dir / "clones"
    cloned_path = clone_repo(url, clones_dir)
    cloned_paths = dict(state.cloned_paths)
    cloned_paths[url] = cloned_path
    return {"cloned_paths": cloned_paths, "repo_index": state.repo_index}


async def explore_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    url = state.repo_urls[state.repo_index - 1]
    clone_path = state.cloned_paths[url]

    result = await explore_repo(
        url=url,
        clone_path=clone_path,
        config=state.config,
        session_dir=state.session_dir,
        userprofile_dir=state.userprofile_dir,
        state=state.session,
        prompts_dir=state.userprofile_dir / "prompts",
    )
    results = list(state.exploration_results)
    results.append(result)
    return {"exploration_results": results}


async def filter_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    projects = _load_projects(state)
    result = await run_filter(
        config=state.config,
        session_dir=state.session_dir,
        state=state.session,
        projects=projects,
        prompts_dir=state.userprofile_dir / "prompts",
    )
    session = state.session.model_copy(deep=True)
    session.selected_projects = result.selected
    return {"filter_result": result, "session": session}


async def writing_node(state: GraphState, kind: str) -> dict[str, Any]:
    assert state.config is not None
    projects = _load_projects(state)
    selected = [p for p in projects if p.title in state.session.selected_projects]

    userinfo_path = state.userprofile_dir / "userinfo.md"
    userinfo_text = userinfo_path.read_text(encoding="utf-8") if userinfo_path.exists() else ""

    result = await write_section(
        kind=kind,
        config=state.config,
        session_dir=state.session_dir,
        state=state.session,
        projects=selected,
        userinfo_text=userinfo_text,
        prompts_dir=state.userprofile_dir / "prompts",
        feedback=state.feedback,
    )

    section = ResumeSection(
        id=result.section_id,
        kind=kind,  # type: ignore
        versions=[SectionVersion(
            content=result.content,
            feedback_used=state.feedback,
            provider="manual" if state.feedback else "ai",
        )],
    )
    session = state.session.model_copy(deep=True)
    existing = [s for s in session.sections if s.id != kind]
    existing.append(section)
    session.sections = existing

    writing_results = dict(state.writing_results)
    writing_results[kind] = result
    return {"session": session, "writing_results": writing_results, "feedback": None}


async def content_critique_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    result = await run_content_critique(
        config=state.config,
        session_dir=state.session_dir,
        state=state.session,
        prompts_dir=state.userprofile_dir / "prompts",
    )
    session = state.session.model_copy(deep=True)
    for issue in result.issues:
        if issue not in session.open_issues:
            session.open_issues.append(issue)
    return {"content_critique_result": result, "session": session}


async def compile_critique_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    tex_path = state.session_dir / "resume.tex"
    result = await run_compile_critique(
        config=state.config,
        session_dir=state.session_dir,
        state=state.session,
        tex_path=tex_path,
        prompts_dir=state.userprofile_dir / "prompts",
    )
    session = state.session.model_copy(deep=True)
    for issue in result.issues:
        if issue not in session.open_issues:
            session.open_issues.append(issue)
    return {"compile_critique_result": result, "session": session}


async def export_node(state: GraphState) -> dict[str, Any]:
    assert state.config is not None
    template_path = state.userprofile_dir / state.config.latex.template_path
    compiler = state.config.latex.compiler
    if compiler == "auto":
        compiler = _detect_latex_compiler() or "pdflatex"

    pdf_path = run_export(
        state=state.session,
        template_path=template_path,
        session_dir=state.session_dir,
        compiler=compiler,
    )
    return {"export_pdf": str(pdf_path)}


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("preflight", preflight_node)
    builder.add_node("clone", clone_node)
    builder.add_node("explore", explore_node)
    builder.add_node("filter", filter_node)
    for kind in SECTION_ORDER:
        builder.add_node(f"write_{kind}", lambda s, k=kind: writing_node(s, k))
    builder.add_node("content_critique", content_critique_node)
    builder.add_node("export", export_node)
    builder.add_node("compile_critique", compile_critique_node)

    builder.set_entry_point("preflight")

    def after_preflight(state: GraphState) -> str:
        return "clone"

    def after_clone(state: GraphState) -> str:
        if state.repo_index >= len(state.repo_urls):
            return "filter"
        return "explore"

    def after_explore(state: GraphState) -> str:
        idx = state.repo_index
        if idx >= len(state.repo_urls):
            return "filter"
        return "clone"

    def after_filter(state: GraphState) -> str:
        return "write_experience"

    writer_edges = {}
    for i, kind in enumerate(SECTION_ORDER):
        if i + 1 < len(SECTION_ORDER):
            writer_edges[f"write_{kind}"] = f"write_{SECTION_ORDER[i + 1]}"
        else:
            writer_edges[f"write_{kind}"] = "content_critique"

    def after_content_critique(state: GraphState) -> str:
        if state.content_critique_result and state.content_critique_result.issues:
            return "export"
        return "export"

    def after_export(state: GraphState) -> str:
        return "compile_critique"

    def after_compile_critique(state: GraphState) -> str:
        return END

    builder.add_conditional_edges("preflight", after_preflight)
    builder.add_conditional_edges("clone", after_clone)
    builder.add_conditional_edges("explore", after_explore)
    builder.add_conditional_edges("filter", after_filter)
    for kind in SECTION_ORDER:
        next_kind = writer_edges[f"write_{kind}"]
        builder.add_edge(f"write_{kind}", next_kind)
    builder.add_conditional_edges("content_critique", after_content_critique)
    builder.add_conditional_edges("export", after_export)
    builder.add_conditional_edges("compile_critique", after_compile_critique)

    conn = sqlite3.connect("userprofile/sessions.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)
