import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

from project_vitae.config import load_config
from project_vitae.graph import build_graph, iter_graph, resume_graph
from project_vitae.io_utils import USERPROFILE_DIR, slugify
from project_vitae.models import SessionState
from project_vitae.session_lock import SessionLock

app = typer.Typer(name="project-vitae")
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def run(
    urls: list[str] = typer.Argument(..., help="GitHub repository URLs"),
    jd: Path = typer.Option(..., "--jd", "-j", help="Path to job description markdown file", exists=True, readable=True),
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

        handler = _HeadlessHandler()
        for event in iter_graph(graph, initial_state.model_dump(), thread_id=session):
            handler.handle(event)

        console.print(f"[green]Done.[/green] Final PDF: {handler.final_pdf or 'N/A'}")
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        lock.release()


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

        handler = _HeadlessHandler(headless=False)
        for event in iter_graph(graph, state.model_dump(), thread_id=session_name):
            handler.handle(event)
        console.print(f"[green]Resume complete.[/green]")
    except Exception as e:
        console.print(f"[red]Resume failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        lock.release()


class _HeadlessHandler:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.final_pdf: str | None = None

    def handle(self, event: dict) -> None:
        for node_name, output in event.items():
            if isinstance(output, dict):
                if "final_pdf" in output and output["final_pdf"]:
                    self.final_pdf = output["final_pdf"]
                if output.get("type") == "filter_confirm":
                    console.print(f"[yellow]Filter proposal:[/yellow] {', '.join(output.get('selected', []))}")
                    if self.headless:
                        output["action"] = "confirm"
                elif output.get("type") == "review":
                    console.print("[yellow]Review interrupted (auto-approving in headless mode)[/yellow]")
                elif output.get("type") == "compile_critique":
                    console.print("[yellow]Compile critique (dismissing in headless mode)[/yellow]")
                    if self.headless:
                        output["action"] = "dismiss"


if __name__ == "__main__":
    app()
