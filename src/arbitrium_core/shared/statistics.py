import statistics
from typing import Literal


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
