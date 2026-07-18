from textual.screen import Screen
from textual.widgets import Header, Label, ListItem, ListView, Static


class ExplorationScreen(Screen):
    def __init__(self, warnings: list[dict], skipped: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warnings = warnings
        self.skipped = skipped

    def compose(self):
        yield Header()
        yield Static("## Repository Exploration", id="title")
        yield Label(f"Skipped repos: {len(self.skipped)}")
        yield ListView(id="results", *self._build_items())

    def _build_items(self) -> list[ListItem]:
        items = []
        for w in self.warnings:
            reason = w.get("reason", "unknown")
            url = w.get("url", "?")
            items.append(ListItem(Label(f"⚠ {url}: {reason}")))
        for s in self.skipped:
            items.append(ListItem(Label(f"❌ {s}")))
        if not items:
            items.append(ListItem(Label("All repos explored successfully")))
        return items
