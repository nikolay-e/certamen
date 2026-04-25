import statistics
from typing import Literal

from certamen.shared.constants import (
    DEFAULT_CONTEXT_SAFETY_MARGIN,
    DEFAULT_MAX_TOKENS_RATIO,
)


def calculate_safe_max_tokens(
    context_window: int,
    max_tokens_ratio: float = DEFAULT_MAX_TOKENS_RATIO,
    safety_margin: float = DEFAULT_CONTEXT_SAFETY_MARGIN,
) -> int:
    safe_context = int(context_window * (1 - safety_margin))
    max_tokens = int(safe_context * max_tokens_ratio)
    return max(1, max_tokens)


def aggregate_scores(
    scores_by_model: dict[str, list[float]],
    method: Literal["mean", "median"] = "mean",
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, scores in scores_by_model.items():
        if not scores:
            continue
        if method == "median":
            result[name] = statistics.median(scores)
        else:
            result[name] = sum(scores) / len(scores)
    return result
