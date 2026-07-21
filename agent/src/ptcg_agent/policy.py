"""Top-level agent policy (Phase 1: heuristic-only, no search yet).

`agent_decide(obs_dict)` is the ONLY entry point `main.py` calls. It must:
  - return the 60-card deck when `obs.select is None` (initial deck request),
  - otherwise return a legal list of option indices for the current selection,
  - NEVER raise: any internal error falls back to `_fallback_decide`, which is trivial
    and cannot itself fail (no dependency on carddb / search / disk / network).

Control flow branches on `SelectType` / `Option.type` ONLY (never `SelectContext`,
which the engine documents as unreliable). Week-2 will route MAIN / trajectory-changing
selects through PIMC search; for now everything is resolved by cheap heuristic rules.
"""
from __future__ import annotations

import os

from cg.api import (
    Observation,
    to_observation_class,
    SelectType,
    OptionType,
)

from . import carddb

_DECK_CACHE: list[int] | None = None
_LOGGED_EXCEPTION = False


# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------
def read_deck_csv() -> list[int]:
    """Read the 60-card deck from deck.csv (cwd, then the Kaggle agent path)."""
    global _DECK_CACHE
    if _DECK_CACHE is not None:
        return list(_DECK_CACHE)

    file_path = "deck.csv"
    if not os.path.exists(file_path):
        # packaged alongside this module, or the Kaggle simulations path.
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
        if os.path.exists(here):
            file_path = here
        else:
            file_path = "/kaggle_simulations/agent/deck.csv"

    with open(file_path, "r") as f:
        rows = f.read().split("\n")
    deck = [int(rows[i]) for i in range(60)]
    _DECK_CACHE = list(deck)
    return deck


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def agent_decide(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    try:
        result = _decide(obs)
        if _is_valid_selection(result, obs):
            return result
        # A malformed heuristic result is a bug — fall through to the safe path.
        return _fallback_decide(obs)
    except Exception:
        _log_exception_once()
        return _fallback_decide(obs)


def _is_valid_selection(sel, obs) -> bool:
    s = obs.select
    if not isinstance(sel, list) or not all(isinstance(i, int) for i in sel):
        return False
    if len(sel) != len(set(sel)):
        return False
    if not (s.minCount <= len(sel) <= s.maxCount):
        return False
    return all(0 <= i < len(s.option) for i in sel)


# ---------------------------------------------------------------------------
# Heuristic dispatch (branch on SelectType only)
# ---------------------------------------------------------------------------
def _decide(obs: Observation) -> list[int]:
    s = obs.select
    st = s.type

    if st == SelectType.MAIN:
        return _decide_main(obs)
    if st == SelectType.ATTACK:
        return _decide_attack(obs)
    if st == SelectType.COUNT:
        return _decide_count(obs)
    # YES_NO, SPECIAL_CONDITION, CARD, ENERGY, EVOLVE, SKILL, etc.:
    # a reasonable generic default until Week-2 gives them dedicated handling.
    return _decide_default(obs)


def _decide_main(obs: Observation) -> list[int]:
    """MAIN phase: set up first, attack when it's worthwhile, otherwise end the turn.

    Priority (each MAIN call makes one move; the engine re-prompts until END):
      attach energy -> evolve -> play a card -> attack (if a good attack is affordable)
      -> use ability -> end. We skip RETREAT/DISCARD here (handled when forced).
    """
    opts = obs.select.option
    by_type: dict[int, list[int]] = {}
    for i, o in enumerate(opts):
        by_type.setdefault(o.type, []).append(i)

    # 1. Attach energy (only ever offered when we still may attach this turn).
    if OptionType.ATTACH in by_type:
        return [by_type[OptionType.ATTACH][0]]
    # 2. Evolve — almost always pure upside.
    if OptionType.EVOLVE in by_type:
        return [by_type[OptionType.EVOLVE][0]]
    # 3. Play a card from hand (draw/search supporters & items build the board).
    if OptionType.PLAY in by_type:
        return [by_type[OptionType.PLAY][0]]
    # 4. Attack if we have an affordable damaging attack on our active.
    if OptionType.ATTACK in by_type:
        idx = _best_main_attack(obs, by_type[OptionType.ATTACK])
        if idx is not None:
            return [idx]
    # 5. Abilities — low priority (some are repeatable; avoid churn in Phase 1).
    if OptionType.ABILITY in by_type:
        return [by_type[OptionType.ABILITY][0]]
    # 6. End the turn.
    if OptionType.END in by_type:
        return [by_type[OptionType.END][0]]

    return _fallback_decide(obs)


def _best_main_attack(obs: Observation, attack_option_idxs: list[int]) -> int | None:
    """From MAIN ATTACK options, pick the affordable one with highest estimated damage."""
    active = _my_active(obs)
    attached = list(active.energies) if active is not None else []
    defender = _opponent_active_card(obs)

    best_idx, best_dmg = None, -1
    for i in attack_option_idxs:
        atk = carddb.ATTACK_BY_ID.get(obs.select.option[i].attackId)
        if atk is None:
            continue
        if attached and not carddb.energy_cost_met(atk, attached):
            continue
        dmg = carddb.compute_damage(atk, defender) if defender is not None else atk.damage
        if dmg > best_dmg:
            best_idx, best_dmg = i, dmg
    # Only attack from MAIN if it actually does damage; otherwise keep setting up / end.
    return best_idx if best_dmg > 0 else None


def _decide_attack(obs: Observation) -> list[int]:
    """SelectType.ATTACK: choose affordable highest-damage attack (fallback: first)."""
    active = _my_active(obs)
    attached = list(active.energies) if active is not None else []
    defender = _opponent_active_card(obs)

    best_idx, best_dmg = None, -1
    for i, o in enumerate(obs.select.option):
        atk = carddb.ATTACK_BY_ID.get(o.attackId)
        if atk is None:
            continue
        affordable = (not attached) or carddb.energy_cost_met(atk, attached)
        dmg = carddb.compute_damage(atk, defender) if defender is not None else atk.damage
        # Prefer affordable + damaging; break ties by damage.
        score = dmg + (1000 if affordable else 0)
        if score > best_dmg:
            best_idx, best_dmg = i, score
    return [best_idx if best_idx is not None else 0]


def _decide_count(obs: Observation) -> list[int]:
    """SelectType.COUNT: pick the option with the largest `number` (e.g. draw as many)."""
    s = obs.select
    best_idx, best_n = 0, None
    for i, o in enumerate(s.option):
        n = o.number if o.number is not None else 0
        if best_n is None or n > best_n:
            best_idx, best_n = i, n
    # COUNT selects a single number option.
    n_pick = max(s.minCount, 1) if s.minCount > 0 else 1
    n_pick = min(n_pick, s.maxCount)
    # Usually maxCount==1 for a number choice; return the best single option.
    return [best_idx] if n_pick >= 1 else []


def _decide_default(obs: Observation) -> list[int]:
    """Generic sensible default: decline optional selects, else take the first minCount.

    minCount==0 means the selection is optional (e.g. optional discard/search) — the
    safe default is to decline. Otherwise satisfy the minimum with the first options.
    """
    s = obs.select
    if s.minCount == 0:
        return []
    return list(range(s.minCount))


# ---------------------------------------------------------------------------
# Fallback — trivial, cannot fail, no external dependencies
# ---------------------------------------------------------------------------
def _fallback_decide(obs: Observation) -> list[int]:
    s = obs.select
    n = s.minCount if s.minCount > 0 else 0
    n = min(n, len(s.option))
    return list(range(n))


def _log_exception_once() -> None:
    global _LOGGED_EXCEPTION
    if not _LOGGED_EXCEPTION:
        _LOGGED_EXCEPTION = True
        # Intentionally silent in the shipped build (no stdout noise / no disk writes).


# ---------------------------------------------------------------------------
# Small observation accessors
# ---------------------------------------------------------------------------
def _my_active(obs: Observation):
    st = obs.current
    if st is None:
        return None
    me = st.players[st.yourIndex]
    return me.active[0] if me.active and me.active[0] is not None else None


def _opponent_active_card(obs: Observation):
    st = obs.current
    if st is None:
        return None
    opp = st.players[1 - st.yourIndex]
    if opp.active and opp.active[0] is not None:
        return carddb.CARD_BY_ID.get(opp.active[0].id)
    return None
