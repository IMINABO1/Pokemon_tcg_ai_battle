"""Smoke test: run one local match, heuristic agent vs a random agent.

Usage:  python harness/smoke_match.py
Run from the `agent/` dir (needs `cg` symlink + `src/` on the path).
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cg.api import to_observation_class  # noqa: E402
from ptcg_agent.policy import agent_decide, read_deck_csv  # noqa: E402
from ptcg_agent import legality  # noqa: E402
from harness.local_match import run_one_match  # noqa: E402


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    s = obs.select
    k = s.minCount if s.minCount > 0 else min(s.maxCount, 1)
    k = min(k, len(s.option))
    return random.sample(range(len(s.option)), k) if k > 0 else []


def main() -> int:
    deck = read_deck_csv()
    print(f"deck: {len(deck)} cards; legal={legality.is_legal(deck)}")
    v = legality.deck_violations(deck)
    if v:
        print("  violations:", v)

    res = run_one_match(deck, agent_decide, deck, random_agent)
    print("\n=== MatchResult ===")
    print(f"winner            : {res.winner}")
    print(f"turns / actions   : {res.turns} / {res.actions}")
    print(f"wall seconds      : {res.wall_seconds:.2f}")
    print(f"max decision (s)  : {res.max_decision_seconds:.4f}")
    if res.error_repr:
        print(f"ERROR (player {res.error_player}):\n{res.error_repr}")
        return 1
    print("no uncaught agent exceptions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
