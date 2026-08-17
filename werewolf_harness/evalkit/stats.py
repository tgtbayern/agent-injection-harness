"""Statistics: intervals, paired comparisons, and sample-size planning.

A point estimate with no interval is not a result. Everything reported goes
through here, and the comparison helpers are paired by seed, because every
configuration is run over the identical seed set -- pairing removes the
between-game variance that otherwise dominates a 60-game batch.

Pure standard library: bootstrap and Wilson intervals need no scipy, and one
fewer dependency is one fewer thing that differs between the machine that ran
the batch and the machine that reads it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    n: int
    method: str

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"

    def to_dict(self) -> dict:
        return {
            "estimate": round(self.estimate, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n": self.n,
            "method": self.method,
        }

    def overlaps(self, other: "Interval") -> bool:
        return not (self.high < other.low or other.high < self.low)


def wilson(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Interval for a proportion. Wilson rather than normal-approximation
    because injection rates live near 0 and 1, where the normal interval runs
    off the end of the scale."""
    if n == 0:
        return Interval(0.0, 0.0, 1.0, 0, "wilson")
    z = Z.get(confidence, 1.96)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n, "wilson")


def bootstrap_mean(
    values: list[float], confidence: float = 0.95, iterations: int = 5000, seed: int = 0
) -> Interval:
    """Percentile bootstrap for a mean. Used for counts and token totals, whose
    distributions are skewed enough that a t-interval misleads."""
    if not values:
        return Interval(0.0, 0.0, 0.0, 0, "bootstrap")
    rng = random.Random(seed)
    n = len(values)
    point = sum(values) / n
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * iterations)]
    hi = means[min(iterations - 1, int((1 + confidence) / 2 * iterations))]
    return Interval(point, lo, hi, n, "bootstrap")


def paired_diff(
    a: dict[int, float], b: dict[int, float], confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Bootstrap interval for the mean paired difference (a - b), keyed by seed.

    Only seeds present in both conditions are used, and the pairing is on the
    seed, not the position in the list -- otherwise a single crashed game
    silently shifts every pair after it.
    """
    common = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in common]
    interval = bootstrap_mean(diffs, confidence=confidence, seed=seed)
    interval.method = "paired_bootstrap"
    return interval


def significant(interval: Interval) -> bool:
    """A paired difference is reportable only if its interval excludes zero."""
    return interval.low > 0 or interval.high < 0


def required_n(sd: float, effect: float, confidence: float = 0.95, power: float = 0.8) -> int:
    """Sample size for a two-sample comparison, the formula the design doc uses:

        n per arm ~= 2 * (z_alpha/2 + z_beta)^2 * sd^2 / effect^2

    Pairing by seed typically halves this again. Run the probe first, put the
    measured sd in here, and let it choose the batch size -- not a round number
    chosen because it looks like enough.
    """
    if effect <= 0:
        raise ValueError("effect size must be positive")
    z_alpha = Z.get(confidence, 1.96)
    z_beta = {0.8: 0.8416, 0.9: 1.2816}.get(power, 0.8416)
    return math.ceil(2 * (z_alpha + z_beta) ** 2 * (sd / effect) ** 2)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
            "min": min(values), "max": max(values)}


__all__ = [
    "Interval",
    "bootstrap_mean",
    "describe",
    "paired_diff",
    "required_n",
    "significant",
    "wilson",
]
