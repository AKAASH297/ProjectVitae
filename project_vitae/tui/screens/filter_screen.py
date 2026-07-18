from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, ListView, ListItem, Static, Header, Footer

from project_vitae.models import FilterResult, ProjectRecord


class FilterScreen(Screen):
    def __init__(self, filter_result: FilterResult, projects: list[ProjectRecord]):
        super().__init__()
        self.filter_result = filter_result
        self.projects = projects

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Filter Results", classes="title"),
            Static(self.filter_result.rationale, id="rationale"),
            Label("Selected Projects:", classes="section_label"),
            ListView(id="project_list"),
            Horizontal(
                Button("Confirm & Continue", id="confirm", variant="success"),
                Button("Go Back", id="back", variant="primary"),
            ),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        list_view = self.query_one("#project_list", ListView)
        for p in self.projects:
            warning = " ⚠️" if p.low_confidence else ""
            icon = "✅" if p.title in self.filter_result.selected else "❌"
            list_view.append(ListItem(Label(f"{icon} {p.title}{warning}")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss({"action": "continue"})
        elif event.button.id == "back":
            self.dismiss({"action": "back"})
