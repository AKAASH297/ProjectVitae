import pytest

from project_vitae.cost import CostGuard, compute_cost
from project_vitae.models import CostCapReached


def test_known_model_cost():
    cost = compute_cost("claude-sonnet-4-20250514", 1000, 500)
    assert cost > 0
    expected_input = (1000 / 1_000_000) * 3.0
    expected_output = (500 / 1_000_000) * 15.0
    assert cost == pytest.approx(expected_input + expected_output)


def test_unknown_model_cost_zero():
    cost = compute_cost("fake-model-v1", 1000, 500)
    assert cost == 0.0


def test_case_insensitive_model():
    cost1 = compute_cost("CLAUDE-SONNET-4", 1000, 500)
    cost2 = compute_cost("claude-sonnet-4", 1000, 500)
    assert cost1 == cost2


def test_overrides_take_precedence():
    overrides = {"gpt-4o": {"input": 1.0, "output": 2.0}}
    cost = compute_cost("gpt-4o", 1_000_000, 500_000, overrides=overrides)
    expected = 1.0 + 1.0
    assert cost == pytest.approx(expected)


def test_zero_tokens():
    cost = compute_cost("claude-sonnet-4", 0, 0)
    assert cost == 0.0


def test_cost_guard_initial():
    g = CostGuard(5.00)
    assert g.current == 0.0


def test_cost_guard_spend_within_cap():
    g = CostGuard(5.00)
    g.spend(1.0)
    assert g.current == 1.0
    g.spend(2.0)
    assert g.current == 3.0


def test_cost_guard_exceeds_cap():
    g = CostGuard(5.00)
    g.spend(4.0)
    with pytest.raises(CostCapReached, match="cost cap"):
        g.spend(2.0)


def test_cost_guard_zero_cost_no_raise():
    g = CostGuard(5.00)
    g.spend(0.0, was_llm=True)
    assert g.current == 0.0


def test_cost_guard_non_llm_excluded_from_cap():
    g = CostGuard(1.00)
    g.spend(5.0, was_llm=False)
    assert g.current == 5.0


def test_cost_guard_reset():
    g = CostGuard(5.00)
    g.spend(3.0)
    g.reset()
    assert g.current == 0.0


def test_cost_guard_edge():
    g = CostGuard(5.00)
    g.spend(5.00)
    with pytest.raises(CostCapReached):
        g.spend(0.01)
