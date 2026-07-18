import logging

from project_vitae.cost import estimate_cost


def test_known_model():
    cost = estimate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    expected = 1000 * 3.0e-6 + 500 * 15.0e-6
    assert abs(cost - expected) < 1e-10


def test_unknown_model_returns_zero(caplog):
    caplog.set_level(logging.WARNING)
    cost = estimate_cost("unknown-model", 1000, 500)
    assert cost == 0.0
    assert "unknown-model" in caplog.text


def test_zero_tokens():
    cost = estimate_cost("gpt-4o", 0, 0)
    assert cost == 0.0


def test_with_overrides():
    overrides = {"my-model": {"input": 5e-6, "output": 10e-6}}
    cost = estimate_cost("my-model", 1000, 500, overrides)
    expected = 1000 * 5e-6 + 500 * 10e-6
    assert abs(cost - expected) < 1e-10


def test_override_extends_known_model():
    overrides = {"gpt-4o": {"input": 1e-6}}
    cost = estimate_cost("gpt-4o", 1000, 0, overrides)
    assert abs(cost - 1000 * 1e-6) < 1e-10


def test_local_provider_zero_cost():
    cost = estimate_cost("ollama/llama2", 1000, 500)
    assert cost == 0.0
