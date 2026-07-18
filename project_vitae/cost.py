import logging

from project_vitae.models import CostCapReached

logger = logging.getLogger(__name__)

PRICING_TABLE: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
}


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, dict[str, float]] | None = None,
) -> float:
    model_lower = model.lower()
    rates = None
    if overrides and model_lower in overrides:
        rates = overrides[model_lower]
    if rates is None and model_lower in PRICING_TABLE:
        rates = PRICING_TABLE[model_lower]
    if rates is None:
        logger.warning("unknown model '%s' — cost treated as 0", model)
        return 0.0
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 6)


class CostGuard:
    def __init__(self, cap_usd: float) -> None:
        self._cap = cap_usd
        self._total = 0.0

    def spend(self, usd: float, was_llm: bool = True) -> None:
        if usd > 0:
            self._total += usd
            if was_llm and self._total > self._cap:
                raise CostCapReached(
                    f"cost cap of ${self._cap:.2f} exceeded (total: ${self._total:.2f})"
                )

    @property
    def current(self) -> float:
        return round(self._total, 6)

    def reset(self) -> None:
        self._total = 0.0
