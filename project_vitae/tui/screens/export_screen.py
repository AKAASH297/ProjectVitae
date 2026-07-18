from textual import on
from textual.screen import Screen
from textual.widgets import Button, Header, Label, Static


class ExportScreen(Screen):
    def __init__(
        self, pdf_path: str | None = None, log_excerpt: str | None = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pdf_path = pdf_path
        self.log_excerpt = log_excerpt

    def compose(self):
        yield Header()
        yield Static("## Export", id="title")
        if self.pdf_path:
            yield Label(f"PDF produced: {self.pdf_path}")
        if self.log_excerpt:
            yield Label(f"LaTeX log:\n{self.log_excerpt}")
        yield Button("Proceed to Compile Critique", id="proceed", variant="primary")

    @on(Button.Pressed, "#proceed")
    def on_proceed(self):
        self.app.pop_screen()
