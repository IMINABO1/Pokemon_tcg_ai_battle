"""Top-level agent policy dispatch with heuristic fast-paths and search routing."""
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from cg.api import (
        Observation, to_observation_class, SelectType, OptionType, SelectContext
    )
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import (
        Observation, to_observation_class, SelectType, OptionType, SelectContext
    )

from .budget import BudgetTracker, budget_for_decision
from .config import LOW_OVERAGE_CUTOFF_SECONDS
from .determinize import StateTracker, OpponentBelief
from .search import search_pimc_action, enumerate_candidate_actions
from . import stats
try:
    from .logging_utils import log_decision
except ImportError:
    def log_decision(*args, **kwargs):
        pass



# Trackers persist across decisions within a process, keyed by our player index so a
# single process driving both seats (self-play in the local harness) keeps each seat's
# belief/bookkeeping separate. In the real Kaggle runtime we are always one fixed index.
_FULL_DECK_IDS: Optional[list[int]] = None
_STATE_TRACKERS: dict[int, StateTracker] = {}
_BELIEF_TRACKERS: dict[int, OpponentBelief] = {}
_DECISION_COUNTS: dict[int, int] = {}


def _reset_game_state() -> None:
    """Clear per-game trackers. Called on the deck-selection prompt, which marks
    a new game; without this, pooled harness workers leak beliefs across games."""
    _STATE_TRACKERS.clear()
    _BELIEF_TRACKERS.clear()
    _DECISION_COUNTS.clear()


def read_deck_csv() -> list[int]:
    """Read deck.csv from current directory or kaggle fallback path."""
    file_path = os.environ.get("PTCG_DECK", "deck.csv")
    if not os.path.exists(file_path):
        pkg_deck = Path(__file__).resolve().parent / "deck.csv"
        if pkg_deck.exists():
            file_path = str(pkg_deck)
        else:
            file_path = "/kaggle_simulations/agent/deck.csv"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().split("\n") if line.strip()]
    return [int(lines[i]) for i in range(min(60, len(lines)))]


def _get_trackers(your_index: int) -> tuple[StateTracker, OpponentBelief]:
    global _FULL_DECK_IDS
    if _FULL_DECK_IDS is None:
        _FULL_DECK_IDS = read_deck_csv()
    if your_index not in _STATE_TRACKERS:
        _STATE_TRACKERS[your_index] = StateTracker(_FULL_DECK_IDS)
        _BELIEF_TRACKERS[your_index] = OpponentBelief()
    return _STATE_TRACKERS[your_index], _BELIEF_TRACKERS[your_index]


def _fallback_decide(obs: Observation) -> list[int]:
    """Trivial fallback selection that cannot fail."""
    if not obs or not obs.select:
        return []
    select = obs.select
    n_options = len(select.option)
    if n_options == 0:
        return []

    min_c = select.minCount
    max_c = select.maxCount

    if min_c == 0 and max_c == 0:
        return []

    count = min_c if min_c > 0 else (1 if max_c > 0 else 0)
    count = min(count, n_options)
    count = min(count, max_c)

    return list(range(count))


def _heuristic_pick(obs: Observation) -> list[int]:
    """Search-free pick for when the time bank is nearly exhausted: attack if
    possible, accept YES prompts, otherwise minimal legal selection."""
    select = obs.select
    if select.minCount <= 1 <= select.maxCount:
        for i, opt in enumerate(select.option):
            if opt.type == OptionType.ATTACK:
                return [i]
        for i, opt in enumerate(select.option):
            if opt.type == OptionType.YES:
                return [i]
    return _fallback_decide(obs)


def _yes_option_index(options) -> int:
    """Index of the YES option (branch on Option.type, never on order)."""
    for i, opt in enumerate(options):
        if opt.type == OptionType.YES:
            return i
    return 0


def _max_number_option_index(options) -> int:
    """Index of the COUNT/NUMBER option with the largest `number` value.

    The engine expects an option index, not the count value itself. Aggressive
    decks generally want the largest available count (draw/place more); the rare
    "discard N" prompts are the accepted downside of keying only on SelectType.
    """
    best_i, best_n = 0, None
    for i, opt in enumerate(options):
        n = opt.number if opt.number is not None else 0
        if best_n is None or n > best_n:
            best_i, best_n = i, n
    return best_i


def _decide(obs: Observation, overage_seconds=None) -> list[int]:
    your_index = obs.current.yourIndex if obs.current else 0
    state_tracker, belief = _get_trackers(your_index)
    belief.update_from_logs(obs)
    _DECISION_COUNTS[your_index] = _DECISION_COUNTS.get(your_index, 0) + 1

    select = obs.select
    stype = select.type
    options = select.option
    n_opts = len(options)

    # Direct fast-path only when there is genuinely no choice; an optional
    # single-target prompt (min=0/max=1, 1 option) is still a real decision
    # and falls through to search.
    if n_opts == 0:
        return []
    if n_opts == 1 and select.minCount >= 1:
        return [0]

    # Fast path for YES_NO prompts (prefer YES for positive actions)
    if stype == SelectType.YES_NO:
        return [_yes_option_index(options)]

    # Fast path for COUNT prompts (e.g. draw count or damage counter count)
    if stype == SelectType.COUNT:
        return [_max_number_option_index(options)]

    # Fast path for SPECIAL_CONDITION
    if stype == SelectType.SPECIAL_CONDITION:
        return [0]

    if overage_seconds is not None and overage_seconds < LOW_OVERAGE_CUTOFF_SECONDS:
        return _heuristic_pick(obs)

    # For MAIN phase and strategic choices, run PIMC search
    budget = BudgetTracker(
        budget_for_decision(overage_seconds, _DECISION_COUNTS.get(your_index, 0))
    )
    action = search_pimc_action(obs, state_tracker, belief, budget)

    log_decision(
        state_turn=obs.current.turn if obs.current else 0,
        your_index=obs.current.yourIndex if obs.current else 0,
        select_type=stype.name if hasattr(stype, "name") else str(stype),
        action_chosen=action,
        score=0.0
    )

    return action


def agent_decide(obs_dict: dict) -> list[int]:
    """Main policy entrypoint."""
    stats.incr("decisions")
    try:
        # remainingOverageTime lives only in the raw dict; to_observation_class
        # drops unknown keys, so read it before converting.
        overage = obs_dict.get("remainingOverageTime") if isinstance(obs_dict, dict) else None
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            _reset_game_state()
            return read_deck_csv()
        return _decide(obs, overage)
    except Exception as e:
        # Exception handler bails safely to fallback
        stats.incr("decision_fallback")
        try:
            obs_obj = to_observation_class(obs_dict) if obs_dict else None
            return _fallback_decide(obs_obj)
        except Exception:
            return []
