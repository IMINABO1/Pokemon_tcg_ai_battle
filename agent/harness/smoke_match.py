"""Smoke test: run one local match, PIMC agent vs the greedy baseline.

Usage:  python agent/harness/smoke_match.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptcg_agent.policy import agent_decide, read_deck_csv  # noqa: E402
from ptcg_agent import legality  # noqa: E402
from harness.local_match import run_one_match, greedy_agent  # noqa: E402


def main() -> int:
    deck = read_deck_csv()
    print(f"deck: {len(deck)} cards; legal={legality.is_legal(deck)}")
    v = legality.deck_violations(deck)
    if v:
        print("  violations:", v)

    res = run_one_match(deck, agent_decide, "PIMC_Agent", deck, greedy_agent, "Greedy_Agent")
    print("\n=== MatchResult ===")
    print(f"winner            : {res.winner} ({res.winner_name})")
    print(f"actions           : {res.total_actions}")
    print(f"wall seconds      : {res.duration_seconds:.2f}")
    print(f"pimc decisions    : {res.stats0.decisions}")
    print(f"pimc max/mean (s) : {res.stats0.max_decision_seconds:.4f} / {res.stats0.mean_decision_seconds:.4f}")
    print(f"pimc total think  : {res.stats0.total_seconds:.1f}s of 600s bank")
    if res.error_message:
        print(f"ERROR: {res.error_message}")
        return 1
    print("no uncaught agent exceptions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
