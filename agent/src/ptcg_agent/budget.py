"""Wall-clock budget tracker for agent decisions."""
import time
from .config import (
    PER_DECISION_BUDGET_SECONDS,
    MIN_DECISION_BUDGET_SECONDS,
    MAX_DECISION_BUDGET_SECONDS,
    EXPECTED_DECISIONS_PER_GAME,
    MIN_EXPECTED_REMAINING_DECISIONS,
)


def budget_for_decision(overage_seconds, decisions_made: int) -> float:
    """Per-decision budget: spread the remaining overage bank over the expected
    remaining decisions. Without an overage reading (local harness), use the
    configured default."""
    if overage_seconds is None:
        return PER_DECISION_BUDGET_SECONDS
    expected_remaining = max(
        MIN_EXPECTED_REMAINING_DECISIONS,
        EXPECTED_DECISIONS_PER_GAME - decisions_made,
    )
    raw = float(overage_seconds) / expected_remaining
    return max(MIN_DECISION_BUDGET_SECONDS, min(MAX_DECISION_BUDGET_SECONDS, raw))


class BudgetTracker:
    def __init__(self, timeout_seconds: float = PER_DECISION_BUDGET_SECONDS):
        self.start_time = time.monotonic()
        self.timeout_seconds = timeout_seconds

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def remaining(self) -> float:
        return max(0.0, self.timeout_seconds - self.elapsed())

    def is_expired(self) -> bool:
        return self.elapsed() >= self.timeout_seconds
