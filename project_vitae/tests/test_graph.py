from unittest.mock import MagicMock

from project_vitae.config import Config, CostConfig, LatexConfig, RetryConfig, SubagentConfig


def _make_cfg() -> MagicMock:
    cfg = MagicMock(spec=Config)
    cfg.latex = LatexConfig(template_path="template.tex", compiler="auto")
    cfg.retry = RetryConfig()
    cfg.cost = CostConfig(per_session_cap_usd=100.0)
    cfg.subagent.side_effect = lambda name: SubagentConfig(
        provider="anthropic",
        api_key_env="TEST_KEY",
        model="claude-sonnet-4-20250514",
        prompt_version="v1",
        temperature=0.3,
        max_tokens=4096,
        system_prompt_override="system",
    )
    return cfg


def test_build_graph_returns_compiled(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")

    cfg = _make_cfg()
    from project_vitae.graph import build_graph

    graph = build_graph(cfg, "test-session")
    assert graph is not None


def test_graph_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xxx")

    cfg = _make_cfg()

    from project_vitae.graph import build_graph

    graph = build_graph(cfg, "test")
    assert graph is not None
