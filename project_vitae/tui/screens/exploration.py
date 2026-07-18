from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Label, ListView, ListItem, Static, Header, Footer

from project_vitae.models import ExplorationResult


class ExplorationScreen(Screen):
    def __init__(
        self,
        urls: list[str],
        userprofile_dir: Path,
        session_dir: Path,
    ):
        super().__init__()
        self.urls = urls
        self.userprofile_dir = userprofile_dir
        self.session_dir = session_dir
        self.results: dict[str, ExplorationResult | str | None] = {}
        for url in urls:
            self.results[url] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Exploring Repositories...", classes="title"),
            ListView(id="repo_list"),
            Static(id="status"),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._render_list()

    def _render_list(self) -> None:
        list_view = self.query_one("#repo_list", ListView)
        list_view.clear()
        for url in self.urls:
            result = self.results.get(url)
            status = "⏳ Pending"
            if result is None:
                status = "⏳ Pending"
            elif isinstance(result, ExplorationResult):
                if result.low_confidence:
                    status = "⚠️ Low confidence"
                else:
                    status = "✅ Done"
            elif result == "skipped":
                status = "⏭️ Skipped"
            elif result == "failed":
                status = "❌ Failed"
            list_view.append(ListItem(Label(f"{url} — {status}")))
