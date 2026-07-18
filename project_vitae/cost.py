HARDCODED_PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0e-6, "output": 15.0e-6},
    "claude-sonnet-4-20250514": {"input": 3.0e-6, "output": 15.0e-6},
    "claude-3-5-sonnet-20241022": {"input": 3.0e-6, "output": 15.0e-6},
    "claude-3-5-haiku-20241022": {"input": 0.8e-6, "output": 4.0e-6},
    "gpt-4o": {"input": 2.5e-6, "output": 10.0e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
}

import logging

logger = logging.getLogger(__name__)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    overrides: dict[str, dict[str, float]] | None = None,
) -> float:
    price_table = HARDCODED_PRICES.copy()
    if overrides:
        for m, prices in overrides.items():
            price_table[m] = {**price_table.get(m, {}), **prices}
    prices = price_table.get(model)
    if prices is None:
        logger.warning("Unknown model '%s' — treating cost as 0", model)
        return 0.0
    return input_tokens * prices["input"] + output_tokens * prices["output"]
