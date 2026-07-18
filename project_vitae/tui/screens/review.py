from textual import on
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, ListView, Static, TextArea

from langgraph.types import Command

from project_vitae.graph import resume_graph


class ReviewScreen(Screen):
    def __init__(self, payload: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload = payload

    def compose(self):
        yield Header()
        yield Static("## Section Review", id="title")
        sections = self.payload.get("sections", [])
        for sec in sections:
            sid = sec.get("id", "?")
            kind = sec.get("kind", "?")
            status = sec.get("status", "draft")
            yield Label(f"{kind} ({sid}) — {status}")
            yield Button(f"Approve {sid}", id=f"approve_{sid}", variant="success")
            yield Button(f"Regenerate {sid}", id=f"regen_{sid}")
        yield Button("Proceed to Export", id="proceed", variant="primary")

    @on(Button.Pressed)
    def on_button(self, event):
        button_id = event.button.id or ""

        if button_id == "proceed":
            approved_ids = [
                s.get("id") for s in self.payload.get("sections", [])
            ]
            self.app.graph_iterator = resume_graph(
                self.app.graph,
                Command(resume={"action": "proceed", "approved_ids": approved_ids}),
                thread_id=self.app.session_name,
            )
            self.app._advance_graph()
            self.app.pop_screen()
        elif button_id.startswith("approve_"):
            pass
        elif button_id.startswith("regen_"):
            section_id = button_id.replace("regen_", "")
            def notify_regen(feedback: str = ""):
                self.app.graph_iterator = resume_graph(
                    self.app.graph,
                    Command(resume={"action": "regen", "section_id": section_id, "feedback": feedback}),
                    thread_id=self.app.session_name,
                )
                self.app._advance_graph()
                self.app.pop_screen()
            notify_regen()
