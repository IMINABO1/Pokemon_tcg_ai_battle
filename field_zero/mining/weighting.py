"""Per-example imitation weights: w = w_skill * w_conf * w_recency * w_diversity.

Applied per (game, player) and inherited by every decision row of that player.
Constants are hyperparameters — tune against BC validation loss, record what
you tried.
"""
from __future__ import annotations
import math

MU_MID = 800.0        # sigmoid midpoint: >= Dragapult-sample-level play trusted
MU_TEMP = 120.0       # sigmoid temperature
SIGMA_C = 0.02        # confidence discount per sigma unit
RECENCY_HALFLIFE_DAYS = 14.0


def w_skill(mu: float | None) -> float:
    if mu is None:
        return 0.25  # unknown rating: keep, but distrust
    return 1.0 / (1.0 + math.exp(-(mu - MU_MID) / MU_TEMP))


def w_confidence(sigma: float | None) -> float:
    if sigma is None:
        return 0.7
    return 1.0 / (1.0 + SIGMA_C * sigma)


def w_recency(age_days: float) -> float:
    return 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)


def w_diversity(archetype_count: int, total: int, n_archetypes: int) -> float:
    """Inverse-frequency, capped. Prevents 50k Alakazam decisions from teaching
    the model that PTCG == Alakazam."""
    if total == 0 or n_archetypes == 0:
        return 1.0
    expected = total / n_archetypes
    ratio = archetype_count / max(expected, 1.0)
    return min(2.0, 1.0 / max(ratio, 0.5))


def combine(mu, sigma, age_days, archetype_count, total, n_archetypes) -> float:
    return (
        w_skill(mu)
        * w_confidence(sigma)
        * w_recency(age_days)
        * w_diversity(archetype_count, total, n_archetypes)
    )
