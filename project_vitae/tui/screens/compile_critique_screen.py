from langgraph.types import Command
from textual import on
from textual.screen import Screen
from textual.widgets import Button, Header, Label, Static

from project_vitae.graph import resume_graph


class CompileCritiqueScreen(Screen):
    def __init__(self, payload: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload = payload

    def compose(self):
        yield Header()
        yield Static("## Compile Critique", id="title")
        issues = self.payload.get("issues", [])
        if issues:
            for iss in issues:
                yield Label(f"  Issue: {iss.get('note', '?')} at {iss.get('location', '?')}")
        else:
            yield Label("No compile issues found.")
        yield Label(f"PDF: {self.payload.get('final_pdf', 'N/A')}")
        yield Button("Dismiss All", id="dismiss", variant="primary")
        yield Button("Request Re-pass", id="repass")

    @on(Button.Pressed, "#dismiss")
    def on_dismiss(self):
        self.app.graph_iterator = resume_graph(
            self.app.graph,
            Command(resume={"action": "dismiss"}),
            thread_id=self.app.session_name,
        )
        self.app._advance_graph()
        self.app.pop_screen()

    @on(Button.Pressed, "#repass")
    def on_repass(self):
        section_ids = [s.get("id", "") for s in self.payload.get("sections", [])]
        self.app.graph_iterator = resume_graph(
            self.app.graph,
            Command(resume={"action": "repass", "section_ids": section_ids, "feedback": ""}),
            thread_id=self.app.session_name,
        )
        self.app._advance_graph()
        self.app.pop_screen()
