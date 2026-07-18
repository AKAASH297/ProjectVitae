from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Label, ListView, ListItem, Header, Footer

from project_vitae.models import CritiqueResult


class CompileCritiqueScreen(Screen):
    def __init__(self, critique: CritiqueResult, pdf_path: str):
        super().__init__()
        self.critique = critique
        self.pdf_path = pdf_path

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Compile Critique Results", classes="title"),
            ListView(id="issue_list"),
            Horizontal(
                Button("Done — PDF Ready", id="done", variant="success"),
                Button("Re-generate Sections", id="regenerate", variant="primary"),
                Button("Dismiss All", id="dismiss", variant="default"),
            ),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        list_view = self.query_one("#issue_list", ListView)
        if not self.critique.issues:
            list_view.append(ListItem(Label("✅ No formatting issues found")))
        else:
            for issue in self.critique.issues:
                list_view.append(ListItem(Label(f"[{issue.location}] {issue.note}")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done":
            self.dismiss({"action": "done", "pdf_path": self.pdf_path})
        elif event.button.id == "regenerate":
            self.dismiss({"action": "regenerate"})
        elif event.button.id == "dismiss":
            self.dismiss({"action": "done", "pdf_path": self.pdf_path})
