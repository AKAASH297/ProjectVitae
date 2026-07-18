from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Label, ListView, ListItem, Header, Footer, Static

from project_vitae.models import CritiqueResult


class ContentCritiqueScreen(Screen):
    def __init__(self, critique: CritiqueResult):
        super().__init__()
        self.critique = critique

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Content Critique Results", classes="title"),
            ListView(id="issue_list"),
            Horizontal(
                Button("Continue to Export", id="continue", variant="success"),
                Button("Re-generate All", id="regenerate", variant="primary"),
            ),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        list_view = self.query_one("#issue_list", ListView)
        for issue in self.critique.issues:
            kw = " [keyword]" if issue.keyword_match else ""
            list_view.append(ListItem(Label(f"[{issue.location}] {issue.note}{kw}")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue":
            self.dismiss({"action": "continue"})
        elif event.button.id == "regenerate":
            self.dismiss({"action": "regenerate"})
