import logging
from pathlib import Path

import typer
from langgraph.types import Command
from rich.console import Console

from project_vitae.config import load_config
from project_vitae.graph import build_graph
from project_vitae.io_utils import USERPROFILE_DIR, slugify
from project_vitae.models import SessionState
from project_vitae.session_lock import SessionLock

app = typer.Typer(name="project-vitae")
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def run(
    urls: list[str] = typer.Argument(..., help="GitHub repository URLs"),
    jd: Path = typer.Option(
        ..., "--jd", "-j", help="Path to job description markdown file", exists=True, readable=True
    ),
    session: str = typer.Option("default", "--session", "-s", help="Session name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force acquire session lock"),
):
    logging.basicConfig(level=logging.INFO)
    try:
        jd_text = jd.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]Error reading JD file:[/red] {e}")
        raise typer.Exit(1)

    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1)

    session_dir = USERPROFILE_DIR / "sessions" / slugify(session)
    try:
        lock = SessionLock(session_dir)
        lock.acquire(force=force)
    except Exception as e:
        console.print(f"[red]Session lock error:[/red] {e}")
        raise typer.Exit(1)

    try:
        graph = build_graph(cfg, session)
        initial_state = SessionState(
            session_name=session,
            github_urls=urls,
            job_description=jd_text,
        )

        thread_config = {"configurable": {"thread_id": session}}

        final_pdf = _run_headless(graph, initial_state.model_dump(), thread_config)
        console.print(f"[green]Done.[/green] Final PDF: {final_pdf or 'N/A'}")
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        lock.release()


def _run_headless(graph, initial_state: dict, config: dict) -> str | None:
    final_pdf: str | None = None

    def _collect_events(stream):
        nonlocal final_pdf
        for event in stream:
            for node_name, output in event.items():
                if isinstance(output, dict):
                    if "final_pdf" in output and output["final_pdf"]:
                        final_pdf = output["final_pdf"]

    _collect_events(graph.stream(initial_state, config, stream_mode="updates"))

    while True:
        state_snapshot = graph.get_state(config)
        next_nodes = state_snapshot.next
        if not next_nodes:
            break

        tasks = state_snapshot.tasks
        resume_value = _resolve_interrupt(tasks)

        _collect_events(graph.stream(Command(resume=resume_value), config, stream_mode="updates"))

    return final_pdf


def _resolve_interrupt(tasks) -> dict:
    for task in tasks:
        interrupts = getattr(task, "interrupts", None)
        if not interrupts:
            continue
        payload = interrupts[0]
        if hasattr(payload, "value"):
            payload = payload.value
        ptype = payload.get("type") if isinstance(payload, dict) else ""

        if ptype == "filter_confirm":
            selected = payload.get("selected", [])
            console.print(
                f"[yellow]Filter:[/yellow] {', '.join(selected) if selected else '(none)'}"
            )
            return {"action": "confirm", "selected": selected}

        elif ptype == "review":
            console.print("[yellow]Review: auto-approving all sections[/yellow]")
            approved_ids = [s["id"] for s in payload.get("sections", [])]
            return {"action": "proceed", "approved_ids": approved_ids}

        elif ptype == "compile_critique":
            console.print("[yellow]Compile critique: dismissing[/yellow]")
            return {"action": "dismiss"}

        else:
            console.print(f"[yellow]Unknown interrupt: {ptype}, dismissing[/yellow]")
            return {}

    return {}


@app.command()
def setup(
    urls: list[str] = typer.Argument(None, help="GitHub repository URLs"),
    jd: Path | None = typer.Option(None, "--jd", "-j", help="Path to job description file"),
    session: str = typer.Option("default", "--session", "-s", help="Session name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    from project_vitae.tui.app import ProjectVitaeApp

    app_tui = ProjectVitaeApp(session_name=session)
    app_tui.run()


@app.command()
def resume(
    session_name: str = typer.Argument(..., help="Session name to resume"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1)

    session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
    if not session_dir.is_dir():
        console.print(f"[red]Session '{session_name}' not found at {session_dir}[/red]")
        raise typer.Exit(1)

    try:
        lock = SessionLock(session_dir)
        lock.acquire_for_resume()
    except Exception as e:
        console.print(f"[red]Session lock error:[/red] {e}")
        raise typer.Exit(1)

    try:
        graph = build_graph(cfg, session_name)
        state_file = session_dir / "resume_state.json"
        if state_file.is_file():
            import json

            state = SessionState.model_validate(json.loads(state_file.read_text()))
        else:
            state = SessionState(session_name=session_name)

        thread_config = {"configurable": {"thread_id": session_name}}
        final_pdf = _run_headless(graph, state.model_dump(), thread_config)
        console.print(f"[green]Resume complete.[/green] Final PDF: {final_pdf or 'N/A'}")
    except Exception as e:
        console.print(f"[red]Resume failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        lock.release()


if __name__ == "__main__":
    app()
