from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, ListView, ListItem, TextArea, Header, Footer, Static

from project_vitae.models import ResumeSection, SessionState


class ReviewScreen(Screen):
    def __init__(self, session_state: SessionState):
        super().__init__()
        self.session_state = session_state
        self._current_section_idx = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Review Sections", classes="title"),
            ListView(id="section_list"),
            Static(id="section_preview"),
            Horizontal(
                Button("Approve", id="approve", variant="success"),
                Button("Regenerate", id="regenerate", variant="primary"),
                Button("Manual Edit", id="edit", variant="default"),
                Button("Diff", id="diff", variant="default"),
            ),
            TextArea(id="feedback_input", text="", classes="hidden"),
            Button("Continue", id="continue", variant="primary"),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._render_list()

    def _render_list(self) -> None:
        list_view = self.query_one("#section_list", ListView)
        list_view.clear()
        for sec in self.session_state.sections:
            status_map = {"draft": "📝", "approved": "✅", "needs_review": "⚠️"}
            icon = status_map.get(sec.status, "📝")
            list_view.append(ListItem(Label(f"{icon} {sec.id} ({sec.status})")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.children.index(event.item)
        self._current_section_idx = idx
        sec = self.session_state.sections[idx]
        self.query_one("#section_preview", Static).update(sec.current.content)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self._update_status("approved")
        elif event.button.id == "regenerate":
            feedback = self.query_one("#feedback_input", TextArea).text
            self.dismiss({"action": "regenerate", "section_idx": self._current_section_idx, "feedback": feedback})
        elif event.button.id == "edit":
            sec = self.session_state.sections[self._current_section_idx]
            self.dismiss({"action": "edit", "section_idx": self._current_section_idx, "content": sec.current.content})
        elif event.button.id == "continue":
            self.dismiss({"action": "continue"})

    def _update_status(self, status: str) -> None:
        if self._current_section_idx < len(self.session_state.sections):
            self.session_state.sections[self._current_section_idx].status = status  # type: ignore
            self._render_list()
