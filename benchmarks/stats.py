"""Statistical utilities for benchmark analysis."""

import random
from math import asin, sqrt
from typing import List, Tuple


def mcnemar(b01: int, b10: int) -> Tuple[float, float]:
    """
    Exact McNemar's test (two-sided) using exact binomial.

    Args:
        b01: baseline correct, arbitrium wrong
        b10: baseline wrong, arbitrium correct

    Returns:
        (chi-square statistic with Yates correction, exact two-sided p-value)
    """
    from math import comb

    n = b01 + b10
    if n == 0:
        return 0.0, 1.0

    # Chi-square with Yates continuity correction (for reference)
    num = (abs(b01 - b10) - 1) ** 2
    chi2 = num / n

    # Exact two-sided p-value via binomial tail with p=0.5
    k = min(b01, b10)
    # P(X <= k) + P(X >= n-k) for two-tailed test
    cdf = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    sf = sum(comb(n, i) for i in range(n - k, n + 1)) / (2**n)
    p_value = min(1.0, 2 * min(cdf, sf))

    return chi2, p_value


def paired_bootstrap_delta_acc(
    y_true: List[str],
    y_a: List[str],
    y_b: List[str],
    iters: int = 10000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap 95% CI for difference in accuracy: acc(A) - acc(B).

    Args:
        y_true: Ground truth labels
        y_a: Predictions from system A
        y_b: Predictions from system B
        iters: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        (mean_delta, (CI_low, CI_high))
    """
    rnd = random.Random(seed)
    n = len(y_true)
    indices = list(range(n))
    diffs = []

    for _ in range(iters):
        sample = [indices[rnd.randrange(n)] for _ in range(n)]
        acc_a = sum(int(y_a[i] == y_true[i]) for i in sample) / n
        acc_b = sum(int(y_b[i] == y_true[i]) for i in sample) / n
        diffs.append(acc_a - acc_b)

    diffs.sort()
    low = diffs[int(0.025 * iters)]
    high = diffs[int(0.975 * iters) - 1]
    mean = sum(diffs) / iters

    return mean, (low, high)


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for difference between two proportions.

    Args:
        p1: Proportion 1 (0.0 to 1.0)
        p2: Proportion 2 (0.0 to 1.0)

    Returns:
        Effect size h (small: 0.2, medium: 0.5, large: 0.8)
    """
    # Clamp to valid range
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))

    return 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))


def compute_cost_normalized_metrics(
    accuracy: float,
    cost_dollars: float,
    duration_seconds: float,
) -> dict[str, float]:
    """
    Compute cost and time normalized metrics.

    Args:
        accuracy: Accuracy as percentage (0-100)
        cost_dollars: Total cost in USD
        duration_seconds: Total duration in seconds

    Returns:
        Dictionary with normalized metrics
    """
    acc_per_dollar = accuracy / cost_dollars if cost_dollars > 0 else float("inf")
    acc_per_minute = accuracy / (duration_seconds / 60) if duration_seconds > 0 else float("inf")

    return {
        "accuracy_per_dollar": acc_per_dollar,
        "accuracy_per_minute": acc_per_minute,
        "cost_per_correct": cost_dollars / accuracy if accuracy > 0 else float("inf"),
        "time_per_correct_seconds": duration_seconds / accuracy if accuracy > 0 else float("inf"),
    }
