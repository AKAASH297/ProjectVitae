from langgraph.types import Command
from textual import on
from textual.screen import Screen
from textual.widgets import Button, Header, Label, Static


class FilterScreen(Screen):
    def __init__(self, payload: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload = payload

    def compose(self):
        yield Header()
        yield Static("## Filter Results", id="title")
        yield Label(f"Selected: {', '.join(self.payload.get('selected', []))}")
        yield Label(f"Rationale: {self.payload.get('rationale', '')}")
        yield Button("Confirm & Continue", id="confirm", variant="primary")
        yield Button("Reject", id="reject", variant="error")

    @on(Button.Pressed, "#confirm")
    def on_confirm(self):
        from project_vitae.graph import resume_graph

        self.app.graph_iterator = resume_graph(
            self.app.graph,
            Command(resume={"action": "confirm", "selected": self.payload.get("selected", [])}),
            thread_id=self.app.session_name,
        )
        self.app._advance_graph()
        self.app.pop_screen()

    @on(Button.Pressed, "#reject")
    def on_reject(self):
        self.notify("Pipeline aborted", severity="error")
        self.app.pop_screen()
