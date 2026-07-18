from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from project_vitae.config import Config, CostConfig, LatexConfig, RetryConfig
from project_vitae.models import ProjectVitaeError, SessionState
from project_vitae.nodes import clone as clone_module
from project_vitae.nodes.clone import make_clone


def _make_state(urls: list[str] | None = None) -> SessionState:
    return SessionState(
        session_name="test-session", github_urls=urls or ["https://github.com/user/repo1"]
    )


def _make_cfg() -> MagicMock:
    cfg = MagicMock(spec=Config)
    cfg.latex = LatexConfig()
    cfg.retry = RetryConfig()
    cfg.cost = CostConfig()
    return cfg


def test_clone_success(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(clone_module, "USERPROFILE_DIR", tmp_path)
    cfg = _make_cfg()

    with (
        patch("project_vitae.nodes.clone._check_github_size", return_value=1000),
        patch("project_vitae.nodes.clone.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        result = make_clone(cfg)(_make_state())

    assert len(result.get("clone_dirs", [])) >= 1
    assert len(result.get("exploration_warnings", [])) == 0


def test_clone_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(clone_module, "USERPROFILE_DIR", tmp_path)
    cfg = _make_cfg()

    with (
        patch("project_vitae.nodes.clone._check_github_size", return_value=1000),
        patch("project_vitae.nodes.clone.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = make_clone(cfg)(_make_state())
        assert len(result.get("clone_dirs", [])) >= 1


def test_clone_all_fail(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(clone_module, "USERPROFILE_DIR", tmp_path)
    cfg = _make_cfg()

    with (
        patch("project_vitae.nodes.clone._check_github_size", return_value=1000),
        patch("project_vitae.nodes.clone.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="fail")
        with pytest.raises(ProjectVitaeError, match="no successful clones"):
            make_clone(cfg)(_make_state())


def test_clone_repo_too_large(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(clone_module, "USERPROFILE_DIR", tmp_path)
    cfg = _make_cfg()

    with patch("project_vitae.nodes.clone._check_github_size", return_value=300 * 1024):
        with pytest.raises(ProjectVitaeError, match="no successful clones"):
            make_clone(cfg)(_make_state())


def test_check_github_size_small(tmp_path: Path, monkeypatch):
    with patch("project_vitae.nodes.clone.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"size": 500}
        mock_get.return_value = mock_response
        from project_vitae.nodes.clone import _check_github_size

        result = _check_github_size("https://github.com/user/repo")
        assert result == 500
