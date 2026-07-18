from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Label, Static, Header, Footer


class ExportScreen(Screen):
    def __init__(self, pdf_path: str | None = None, error: str | None = None):
        super().__init__()
        self._pdf_path = pdf_path
        self._error = error

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Export", classes="title"),
            Static(id="export_status"),
            Horizontal(
                Button("Open PDF", id="open_pdf", variant="success"),
                Button("Back to Review", id="back", variant="primary"),
                Button("Done", id="done", variant="default"),
            ),
            id="main_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        status = self.query_one("#export_status", Static)
        if self._pdf_path:
            status.update(f"✅ PDF produced at:\n{self._pdf_path}")
        elif self._error:
            status.update(f"❌ Export failed:\n{self._error}")
        else:
            status.update("Exporting...")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done":
            self.dismiss({"action": "done"})
        elif event.button.id == "back":
            self.dismiss({"action": "back"})
        elif event.button.id == "open_pdf" and self._pdf_path:
            import subprocess
            subprocess.Popen(["start", self._pdf_path], shell=True)
