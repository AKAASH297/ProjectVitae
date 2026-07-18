from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, ListView, ListItem, TextArea, Header, Footer

from project_vitae.config import AppConfig
from project_vitae.session_lock import has_lock


class SetupScreen(Screen):
    def __init__(
        self,
        config: AppConfig,
        userprofile_dir: Path,
        session_dir: Path,
    ):
        super().__init__()
        self.app_config = config
        self.userprofile_dir = userprofile_dir
        self.session_dir = session_dir
        self._sessions_dir = userprofile_dir / "sessions"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("ProjectVitae — Resume Builder", classes="title"),
            Vertical(
                Label("GitHub Repository URLs (one per line):"),
                TextArea(id="urls", text=""),
                Label("Session Name:"),
                Input(id="session_name", value=self.session_dir.name, placeholder="e.g. my-resume"),
                Label("Paused/Abandoned Sessions:"),
                ListView(id="session_list"),
                Horizontal(
                    Button("Resume Selected", id="resume", variant="primary"),
                    Button("Discard Selected", id="discard", variant="error"),
                    Button("Start New Session", id="start", variant="success"),
                ),
                id="setup_content",
            ),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_session_list()

    def _refresh_session_list(self) -> None:
        list_view = self.query_one("#session_list", ListView)
        list_view.clear()
        if self._sessions_dir.exists():
            for child in self._sessions_dir.iterdir():
                if child.is_dir() and (child / "resume_state.json").exists():
                    locked = " [locked]" if has_lock(child) else ""
                    list_view.append(ListItem(Label(f"{child.name}{locked}")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self._start_session()
        elif event.button.id == "resume":
            self._resume_session()
        elif event.button.id == "discard":
            self._discard_session()

    def _start_session(self) -> None:
        urls_text = self.query_one("#urls", TextArea).text
        session_name = self.query_one("#session_name", Input).value

        if not urls_text.strip():
            self.notify("Please enter at least one URL", severity="error")
            return
        if not session_name.strip():
            self.notify("Please enter a session name", severity="error")
            return

        urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
        self.dismiss({
            "action": "start",
            "urls": urls,
            "session_name": session_name,
        })

    def _resume_session(self) -> None:
        list_view = self.query_one("#session_list", ListView)
        if list_view.index is not None:
            item = list_view.children[list_view.index]
            name = item.query(Label).first().renderable
            name = name.replace(" [locked]", "")
            self.dismiss({
                "action": "resume",
                "session_name": name,
            })

    def _discard_session(self) -> None:
        list_view = self.query_one("#session_list", ListView)
        if list_view.index is not None:
            item = list_view.children[list_view.index]
            name = item.query(Label).first().renderable
            name = name.replace(" [locked]", "")
            sess_path = self._sessions_dir / name
            import shutil
            shutil.rmtree(sess_path, ignore_errors=True)
            self._refresh_session_list()
            self.notify(f"Session '{name}' discarded")
