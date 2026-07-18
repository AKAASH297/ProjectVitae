from pathlib import Path

from textual import on
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, ListItem, ListView, Static, TextArea

from project_vitae.io_utils import USERPROFILE_DIR
from project_vitae.session_lock import list_resumable_sessions


class SetupScreen(Screen):
    def compose(self):
        yield Header()
        yield Static("## Setup", id="title")
        yield Label("GitHub Repository URLs (one per line):")
        yield TextArea(id="urls", language="", text="")
        yield Label("Job Description:")
        yield TextArea(id="jd", language="markdown", text="")
        yield Label("Session Name:")
        yield Input(id="session_name", placeholder="my-resume-session")
        yield Button("Start Pipeline", id="start", variant="primary")
        yield Static("## Resumable Sessions", id="resumable_title")
        yield ListView(id="resumable_list", *self._load_resumable())

    def _load_resumable(self) -> list[ListItem]:
        items = []
        for name in list_resumable_sessions():
            items.append(ListItem(Label(f"Resume: {name}")))
        if not items:
            items.append(ListItem(Label("No paused sessions")))
        return items

    @on(Button.Pressed, "#start")
    def on_start(self):
        urls_text = self.query_one("#urls", TextArea).text.strip()
        jd = self.query_one("#jd", TextArea).text.strip()
        session_name = self.query_one("#session_name", Input).value.strip()

        if not urls_text:
            self.notify("Enter at least one GitHub URL", severity="error")
            return
        if not jd:
            self.notify("Enter a job description", severity="error")
            return
        if not session_name:
            session_name = "default"

        urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
        self.app.start_pipeline(urls, jd, session_name)

    @on(ListView.Selected, "#resumable_list")
    def on_resume(self, event):
        label = str(event.item.children[0].renderable)
        if "Resume:" in label:
            name = label.replace("Resume:", "").strip()
            self.app.resume_pipeline(name)
