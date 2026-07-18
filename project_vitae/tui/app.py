from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.screen import Screen

from project_vitae.config import AppConfig
from project_vitae.tui.screens.setup import SetupScreen


class ProjectVitaeApp(App):
    SCREENS: dict[str, Screen] = {}

    def __init__(
        self,
        config: AppConfig,
        userprofile_dir: Path,
        session_dir: Path,
    ):
        super().__init__()
        self.app_config = config
        self.userprofile_dir = userprofile_dir
        self.session_dir = session_dir

    def on_mount(self) -> None:
        self.push_screen(SetupScreen(
            config=self.app_config,
            userprofile_dir=self.userprofile_dir,
            session_dir=self.session_dir,
        ))


async def run_tui(
    config: AppConfig,
    userprofile_dir: Path,
    session_dir: Path,
) -> None:
    app = ProjectVitaeApp(config, userprofile_dir, session_dir)
    await app.run_async()
