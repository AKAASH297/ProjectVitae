from typer.testing import CliRunner

from project_vitae.__main__ import app

runner = CliRunner()


def test_run_missing_jd():
    result = runner.invoke(app, ["run", "https://github.com/user/repo"])
    assert result.exit_code != 0


def test_run_no_urls():
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_run_bad_jd_path(tmp_path):
    bad_path = tmp_path / "nonexistent.md"
    result = runner.invoke(
        app,
        [
            "run",
            "https://github.com/user/repo",
            "--jd",
            str(bad_path),
        ],
    )
    assert result.exit_code != 0
