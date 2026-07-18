import logging
from typing import Any, Iterator, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from project_vitae.config import Config
from project_vitae.models import ProjectVitaeError, SessionState

from .nodes.clone import make_clone
from .nodes.compile_critique import make_compile_critique
from .nodes.content_critique import make_content_critique
from .nodes.explore import make_explore
from .nodes.export_node import make_export
from .nodes.filter_node import make_filter
from .nodes.preflight import make_preflight
from .nodes.writing import SECTION_ORDER, make_writing

logger = logging.getLogger(__name__)


def build_graph(cfg: Config, session_name: str) -> CompiledStateGraph:
    workflow = StateGraph(SessionState)

    workflow.add_node("preflight", make_preflight(cfg))
    workflow.add_node("clone", make_clone(cfg))

    def explore_loop(state: SessionState) -> dict:
        clone_dirs = state.clone_dirs
        github_urls = state.github_urls
        skipped = list(state.skipped_repos)
        warnings = list(state.exploration_warnings)

        for i, url in enumerate(github_urls):
            if url in skipped:
                continue
            clone_dir = clone_dirs[i] if i < len(clone_dirs) else ""
            if not clone_dir:
                continue

            explore_result = make_explore(cfg, url, clone_dir)(state)
            if explore_result.get("current_exploration") is None:
                new_skipped = explore_result.get("skipped_repos", [])
                skipped = list(set(skipped + new_skipped))
                new_warnings = explore_result.get("exploration_warnings", [])
                warnings.extend(new_warnings)
            else:
                state = SessionState(**{**state.model_dump(), **explore_result})

        return {
            "skipped_repos": skipped,
            "exploration_warnings": warnings,
        }

    workflow.add_node("explore_loop", explore_loop)
    workflow.add_node("filter", make_filter(cfg))

    def filter_pause(state: SessionState) -> dict:
        proposal = state.filter_proposal
        if proposal is None:
            raise ProjectVitaeError("filter did not produce a proposal — cannot continue")

        payload = {
            "type": "filter_confirm",
            "selected": proposal.selected,
            "rationale": proposal.rationale,
        }

        result = interrupt(payload)

        if isinstance(result, dict):
            action = result.get("action", "confirm")
            if action in ("confirm", "proceed"):
                selected = result.get("selected", proposal.selected)
                return {"selected_projects": selected}
            if action == "reject":
                raise ProjectVitaeError("filter rejected — pipeline aborted")
        raise ProjectVitaeError("invalid filter interrupt response")

    workflow.add_node("filter_pause", filter_pause)

    writing_nodes = {}
    for kind in SECTION_ORDER:
        node_name = f"writing_{kind}"
        workflow.add_node(node_name, make_writing(cfg, kind))
        writing_nodes[kind] = node_name

    workflow.add_node("content_critique", make_content_critique(cfg))

    def review_pause(state: SessionState) -> dict:
        payload = {
            "type": "review",
            "sections": [
                {"id": s.id, "kind": s.kind, "status": s.status, "content": s.current.content}
                for s in state.sections
            ],
            "issues": [i.model_dump() for i in state.open_issues],
        }

        result = interrupt(payload)

        if isinstance(result, dict):
            action = result.get("action", "proceed")
            if action == "proceed":
                approved_ids = result.get("approved_ids", [s.id for s in state.sections])
                sections = list(state.sections)
                for s in sections:
                    if s.id in approved_ids:
                        s.status = "approved"
                return {"sections": sections}
            elif action == "regen":
                section_id = result.get("section_id", "")
                feedback = result.get("feedback", "")
                return {
                    "current_section_kind": _kind_from_id(state, section_id),
                    "current_feedback": feedback,
                }
            elif action == "manual_edit":
                section_id = result.get("section_id", "")
                new_content = result.get("content", "")
                return _apply_manual_edit(state, section_id, new_content)
        raise ProjectVitaeError("invalid review interrupt response")

    workflow.add_node("review_pause", review_pause)

    workflow.add_node("export", make_export(cfg))

    workflow.add_node("compile_critique", make_compile_critique(cfg))

    def compile_pause(state: SessionState) -> dict:
        payload = {
            "type": "compile_critique",
            "issues": [i.model_dump() for i in state.open_issues if i.phase == "compile"],
            "final_pdf": state.final_pdf,
        }

        result = interrupt(payload)

        if isinstance(result, dict):
            action = result.get("action", "dismiss")
            if action == "dismiss":
                return {}
            elif action == "repass":
                section_ids = result.get("section_ids", [s.id for s in state.sections])
                feedback = result.get("feedback", "")
                if section_ids:
                    return {
                        "current_section_kind": _kind_from_id(state, section_ids[0]),
                        "current_feedback": feedback,
                    }
                return {}
        raise ProjectVitaeError("invalid compile critique interrupt response")

    workflow.add_node("compile_pause", compile_pause)

    workflow.set_entry_point("preflight")

    workflow.add_edge("preflight", "clone")
    workflow.add_edge("clone", "explore_loop")
    workflow.add_edge("explore_loop", "filter")
    workflow.add_edge("filter", "filter_pause")

    workflow.add_conditional_edges(
        "filter_pause",
        lambda s: "writing_experience" if s.selected_projects else END,
    )

    prev = "filter_pause"
    for kind in SECTION_ORDER:
        node_name = writing_nodes[kind]
        workflow.add_edge(prev, node_name)
        prev = node_name

    workflow.add_edge(prev, "content_critique")
    workflow.add_edge("content_critique", "review_pause")

    def review_router(
        state: SessionState,
    ) -> Literal[
        "writing_experience", "writing_education", "writing_skills", "writing_summary", "export"
    ]:
        kind = state.current_section_kind
        if kind and kind in SECTION_ORDER:
            return writing_nodes[kind]
        return "export"

    workflow.add_conditional_edges(
        "review_pause",
        review_router,
        {
            writing_nodes["experience"]: writing_nodes["experience"],
            writing_nodes["education"]: writing_nodes["education"],
            writing_nodes["skills"]: writing_nodes["skills"],
            writing_nodes["summary"]: writing_nodes["summary"],
            "export": "export",
        },
    )

    workflow.add_edge("export", "compile_critique")
    workflow.add_edge("compile_critique", "compile_pause")

    def compile_router(
        state: SessionState,
    ) -> Literal[
        "writing_experience", "writing_education", "writing_skills", "writing_summary", END
    ]:
        kind = state.current_section_kind
        if kind and kind in SECTION_ORDER:
            return writing_nodes[kind]
        return END

    workflow.add_conditional_edges(
        "compile_pause",
        compile_router,
        {
            writing_nodes["experience"]: writing_nodes["experience"],
            writing_nodes["education"]: writing_nodes["education"],
            writing_nodes["skills"]: writing_nodes["skills"],
            writing_nodes["summary"]: writing_nodes["summary"],
            END: END,
        },
    )

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


def iter_graph(
    graph: CompiledStateGraph,
    initial_state: dict[str, Any],
    thread_id: str = "default",
) -> Iterator[dict[str, Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        yield event


def resume_graph(
    graph: CompiledStateGraph,
    command: Command,
    thread_id: str = "default",
) -> Iterator[dict[str, Any]]:
    config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream(command, config, stream_mode="updates"):
        yield event


def _kind_from_id(state: SessionState, section_id: str) -> str:
    for s in state.sections:
        if s.id == section_id:
            return s.kind
    return "experience"


def _apply_manual_edit(state: SessionState, section_id: str, new_content: str) -> dict:
    from datetime import datetime, timezone

    from project_vitae.models import SectionVersion

    sections = list(state.sections)
    for s in sections:
        if s.id == section_id:
            s.versions.append(
                SectionVersion(
                    content=new_content,
                    timestamp=datetime.now(timezone.utc),
                    provider="manual",
                )
            )
            s.status = "draft"
    return {"sections": sections, "current_section_kind": None, "current_feedback": None}
