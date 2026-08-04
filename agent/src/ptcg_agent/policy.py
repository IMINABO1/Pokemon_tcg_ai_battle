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

from .budget import BudgetTracker
from .determinize import StateTracker, OpponentBelief
from .search import search_pimc_action, enumerate_candidate_actions
from . import stats
try:
    from .logging_utils import log_decision
except ImportError:
    def log_decision(*args, **kwargs):
        pass



# Global state persistent across decisions within a process
_FULL_DECK_IDS: Optional[list[int]] = None
_STATE_TRACKER: Optional[StateTracker] = None
_BELIEF_TRACKER: Optional[OpponentBelief] = None


def read_deck_csv() -> list[int]:
    """Read deck.csv from current directory or kaggle fallback path."""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        pkg_deck = Path(__file__).resolve().parent / "deck.csv"
        if pkg_deck.exists():
            file_path = str(pkg_deck)
        else:
            file_path = "/kaggle_simulations/agent/deck.csv"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().split("\n") if line.strip()]
    return [int(lines[i]) for i in range(min(60, len(lines)))]


def _init_trackers():
    global _FULL_DECK_IDS, _STATE_TRACKER, _BELIEF_TRACKER
    if _FULL_DECK_IDS is None:
        _FULL_DECK_IDS = read_deck_csv()
        _STATE_TRACKER = StateTracker(_FULL_DECK_IDS)
        _BELIEF_TRACKER = OpponentBelief()


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


def _decide(obs: Observation) -> list[int]:
    _init_trackers()
    _BELIEF_TRACKER.update_from_logs(obs)

    select = obs.select
    stype = select.type
    options = select.option
    n_opts = len(options)

    # Direct fast-path for trivial single choices
    if n_opts <= 1 and select.minCount <= 1:
        return enumerate_candidate_actions(obs)[0]

    # Fast path for YES_NO prompts (prefer YES for positive actions)
    if stype == SelectType.YES_NO:
        # Default YES (option 0)
        return [0]

    # Fast path for COUNT prompts (e.g. draw count or damage counter count)
    if stype == SelectType.COUNT:
        # Choose max count
        return [select.maxCount]

    # Fast path for SPECIAL_CONDITION
    if stype == SelectType.SPECIAL_CONDITION:
        return [0]

    # For MAIN phase and strategic choices, run PIMC search
    budget = BudgetTracker()
    action = search_pimc_action(obs, _STATE_TRACKER, _BELIEF_TRACKER, budget)

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
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        return _decide(obs)
    except Exception as e:
        # Exception handler bails safely to fallback
        stats.incr("decision_fallback")
        try:
            obs_obj = to_observation_class(obs_dict) if obs_dict else None
            return _fallback_decide(obs_obj)
        except Exception:
            return []
