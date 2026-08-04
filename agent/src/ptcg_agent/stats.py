"""In-process counters that make silently-swallowed failures observable.

Phase 0 instrumentation: the agent's search failures were caught and discarded
with no record, so a search that never ran looked identical to one that did.
These counters are incremented at each swallow site; the harness resets them
before a match and snapshots them after, surfacing the numbers into the results
CSV. No control flow depends on them.
"""

_COUNTERS: dict[str, int] = {}


def reset() -> None:
    _COUNTERS.clear()


def incr(key: str, n: int = 1) -> None:
    _COUNTERS[key] = _COUNTERS.get(key, 0) + n


def snapshot() -> dict[str, int]:
    return dict(_COUNTERS)
