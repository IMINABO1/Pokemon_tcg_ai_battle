"""Wall-clock budget tracker for agent decisions."""
import time
from .config import PER_DECISION_BUDGET_SECONDS


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
