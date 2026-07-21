"""Local match driver, built directly on cg.game (no Kaggle runner needed).

`Battle.battle_ptr` is a process-level singleton, so exactly ONE match runs per Python
process. To run many matches concurrently, spawn one process per match (see run_matches.py)
— never call run_one_match twice in the same process without a fresh interpreter.

The driving loop is intentionally reusable as the skeleton for a future self-play
data-generation harness; only the per-decision logging payload would change.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from cg import game
from cg.api import to_observation_class

# An agent is a callable: obs_dict -> list[int] (same contract as Kaggle's `agent`).
AgentFn = Callable[[dict], list[int]]


@dataclass
class MatchResult:
    winner: int | None            # 0, 1, 2(draw), or None if unresolved/errored
    reason: int | None            # RESULT.reason code from the engine, if known
    turns: int
    actions: int
    wall_seconds: float
    error_player: int | None = None      # player whose agent raised (forced loss)
    error_repr: str | None = None
    max_decision_seconds: float = 0.0
    fallback_note: str = ""
    logs_tail: list = field(default_factory=list)


def run_one_match(
    deck0: list[int],
    agent0: AgentFn,
    deck1: list[int],
    agent1: AgentFn,
    max_actions: int = 3000,
) -> MatchResult:
    """Drive one full game to completion. Always calls battle_finish() in finally."""
    agents = [agent0, agent1]
    start = time.monotonic()
    actions = 0
    max_decision = 0.0
    winner: int | None = None
    reason: int | None = None
    error_player: int | None = None
    error_repr: str | None = None
    last_state = None

    obs, start_data = game.battle_start(deck0, deck1)
    try:
        if obs is None:
            # Engine rejected the decks (e.g. illegal). Surface the error codes.
            return MatchResult(
                winner=None, reason=None, turns=0, actions=0,
                wall_seconds=time.monotonic() - start,
                error_repr=f"battle_start failed: errorPlayer={start_data.errorPlayer} "
                           f"errorType={start_data.errorType}",
            )

        while actions < max_actions:
            o = to_observation_class(obs)
            last_state = o.current

            # Match finished?
            if o.current is not None and o.current.result != -1:
                winner = o.current.result
                break
            if o.select is None:
                # Only expected at the very first deck request; the engine drives
                # deck submission internally in local play, so treat as terminal.
                break

            player = o.current.yourIndex if o.current is not None else 0
            t0 = time.monotonic()
            try:
                sel = agents[player](obs)
            except Exception:  # noqa: BLE001 — an agent crash is a forced loss, not ours
                error_player = player
                error_repr = traceback.format_exc(limit=4)
                winner = 1 - player
                break
            max_decision = max(max_decision, time.monotonic() - t0)

            try:
                obs = game.battle_select(sel)
            except Exception:  # noqa: BLE001 — illegal selection => forced loss
                error_player = player
                error_repr = f"illegal select {sel}: {traceback.format_exc(limit=3)}"
                winner = 1 - player
                break
            actions += 1
    finally:
        game.battle_finish()

    turns = last_state.turn if last_state is not None else 0
    logs_tail = []
    if last_state is None and obs is not None:
        try:
            logs_tail = to_observation_class(obs).logs[-5:]
        except Exception:  # noqa: BLE001
            pass

    return MatchResult(
        winner=winner, reason=reason, turns=turns, actions=actions,
        wall_seconds=time.monotonic() - start,
        error_player=error_player, error_repr=error_repr,
        max_decision_seconds=max_decision, logs_tail=logs_tail,
    )
