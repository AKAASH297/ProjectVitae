import logging
from pathlib import Path
from typing import Any

from textual.app import App
from textual.binding import Binding

from project_vitae.config import Config, load_config
from project_vitae.graph import build_graph, iter_graph, resume_graph
from project_vitae.io_utils import USERPROFILE_DIR, slugify
from project_vitae.models import SessionState
from project_vitae.session_lock import SessionLock, list_resumable_sessions

from .screens.setup import SetupScreen
from .screens.exploration import ExplorationScreen
from .screens.filter_screen import FilterScreen
from .screens.review import ReviewScreen
from .screens.export_screen import ExportScreen
from .screens.compile_critique_screen import CompileCritiqueScreen

logger = logging.getLogger(__name__)


class ProjectVitaeApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+s", "save_state", "Save"),
    ]

    def __init__(self, session_name: str | None = None):
        super().__init__()
        self.session_name = session_name or ""
        self.cfg: Config | None = None
        self.graph = None
        self.graph_iterator: Any = None
        self._lock: SessionLock | None = None

    def on_mount(self) -> None:
        self.push_screen(SetupScreen())

    def start_pipeline(self, urls: list[str], jd: str, session_name: str) -> None:
        self.session_name = session_name
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
        self._lock = SessionLock(session_dir)
        self._lock.acquire()

        self.cfg = load_config()
        self.graph = build_graph(self.cfg, session_name)

        initial_state = SessionState(
            session_name=session_name,
            github_urls=urls,
            job_description=jd,
        )

        self.graph_iterator = iter_graph(
            self.graph,
            initial_state.model_dump(),
            thread_id=session_name,
        )

        self._advance_graph()

    def resume_pipeline(self, session_name: str) -> None:
        self.session_name = session_name
        session_dir = USERPROFILE_DIR / "sessions" / slugify(session_name)
        self._lock = SessionLock(session_dir)
        self._lock.acquire_for_resume()

        self.cfg = load_config()
        self.graph = build_graph(self.cfg, session_name)

    def _advance_graph(self) -> None:
        import threading
        def run():
            try:
                for event in (self.graph_iterator or []):
                    self.call_from_thread(self._handle_event, event)
            except Exception as e:
                logger.error("pipeline error: %s", e, exc_info=True)
                self.call_from_thread(self._show_error, str(e))
        threading.Thread(target=run, daemon=True).start()

    def _handle_event(self, event: dict[str, Any]) -> None:
        for node_name, output in event.items():
            if isinstance(output, dict) and "type" in output:
                self._handle_interrupt(output)

    def _handle_interrupt(self, payload: dict) -> None:
        ptype = payload.get("type", "")
        if ptype == "filter_confirm":
            self.push_screen(FilterScreen(payload))
        elif ptype == "review":
            self.push_screen(ReviewScreen(payload))
        elif ptype == "compile_critique":
            self.push_screen(CompileCritiqueScreen(payload))

    def _show_error(self, message: str) -> None:
        from textual.screen import ModalScreen
        class ErrorScreen(ModalScreen[None]):
            def compose(self):
                from textual.widgets import Label, Button
                yield Label(message)
                yield Button("OK")
            def on_button_pressed(self, event):
                self.app.pop_screen()
        self.push_screen(ErrorScreen())

    def action_save_state(self) -> None:
        pass
