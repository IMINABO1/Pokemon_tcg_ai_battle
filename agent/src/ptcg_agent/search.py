"""PIMC Search Engine with shallow expectimax rollout and root averaging."""
import random
import sys
from pathlib import Path
from typing import Callable, Optional

try:
    from cg.api import (
        Observation, SearchState, search_begin, search_step, search_end, search_release,
        SelectType, OptionType, State
    )
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import (
        Observation, SearchState, search_begin, search_step, search_end, search_release,
        SelectType, OptionType, State
    )

from .config import NUM_DETERMINIZATIONS, MAX_SEARCH_DEPTH, MAX_ACTION_CANDIDATES
from .budget import BudgetTracker
from .evaluate import evaluate_state
from .determinize import StateTracker, OpponentBelief, sample_determinization
from . import stats


def enumerate_candidate_actions(obs: Observation) -> list[list[int]]:
    """Enumerate valid action option index lists for a decision prompt."""
    if not obs or not obs.select:
        return []

    select = obs.select
    n_options = len(select.option)
    if n_options == 0:
        return [[]]

    min_c = select.minCount
    max_c = select.maxCount

    # Simple 1-option selection
    if min_c == 1 and max_c == 1:
        # Cap candidates if too many
        if n_options > MAX_ACTION_CANDIDATES:
            # Shortlist first MAX_ACTION_CANDIDATES options
            return [[i] for i in range(MAX_ACTION_CANDIDATES)]
        return [[i] for i in range(n_options)]

    # 0 or 1 selection (e.g. optional action)
    if min_c == 0 and max_c == 1:
        candidates = [[]]
        limit = min(n_options, MAX_ACTION_CANDIDATES - 1)
        candidates.extend([[i] for i in range(limit)])
        return candidates

    # Select exactly maxCount options (e.g. setup active/bench or multi-card select)
    if min_c == max_c:
        if n_options <= max_c:
            return [list(range(n_options))]
        # Sample or take first max_c
        return [list(range(max_c))]

    # Default fallback: pick maxCount elements
    limit_c = min(max_c, n_options)
    if limit_c == 0:
        return [[]]
    return [list(range(limit_c))]


def _rollout(
    start_search_state: SearchState,
    initial_action: list[int],
    depth: int,
    your_index: int,
    evaluator: Callable[[State, int], float]
) -> float:
    """Step forward with initial_action and greedy rollout to leaf depth."""
    current_search_state = search_step(start_search_state.searchId, initial_action)

    current_depth = 1
    while current_depth < depth:
        obs = current_search_state.observation
        if not obs or not obs.select or (obs.current and obs.current.result != -1):
            break

        candidates = enumerate_candidate_actions(obs)
        if not candidates:
            break

        # Pick first/default candidate for greedy rollout
        next_action = candidates[0]
        try:
            current_search_state = search_step(current_search_state.searchId, next_action)
        except Exception:
            break
        current_depth += 1

    leaf_obs = current_search_state.observation
    if leaf_obs and leaf_obs.current:
        score = evaluator(leaf_obs.current, your_index)
    else:
        score = 0.0

    return score


def search_pimc_action(
    obs: Observation,
    state_tracker: StateTracker,
    belief: OpponentBelief,
    budget: BudgetTracker,
    evaluator: Callable[[State, int], float] = evaluate_state
) -> list[int]:
    """PIMC root-averaging search over sampled determinizations.
    
    Returns:
        Best action (list[int] option indices).
    """
    candidates = enumerate_candidate_actions(obs)
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates[0]

    your_index = obs.current.yourIndex if obs.current else 0
    action_scores: dict[tuple[int, ...], float] = {tuple(c): 0.0 for c in candidates}
    action_counts: dict[tuple[int, ...], int] = {tuple(c): 0 for c in candidates}

    determinizations_run = 0

    stats.incr("search_decisions")

    while determinizations_run < NUM_DETERMINIZATIONS and not budget.is_expired():
        det = sample_determinization(obs, state_tracker, belief)

        try:
            root_state = search_begin(
                obs,
                your_deck=det["your_deck"],
                your_prize=det["your_prize"],
                opponent_deck=det["opponent_deck"],
                opponent_prize=det["opponent_prize"],
                opponent_hand=det["opponent_hand"],
                opponent_active=det["opponent_active"],
            )
        except Exception:
            # If search_begin rejects determinization, continue to next
            stats.incr("search_begin_fail")
            determinizations_run += 1
            continue

        stats.incr("search_begin_ok")

        try:
            for cand in candidates:
                if budget.is_expired():
                    break
                cand_tuple = tuple(cand)
                stats.incr("candidates_total")
                try:
                    score = _rollout(
                        root_state,
                        cand,
                        MAX_SEARCH_DEPTH,
                        your_index,
                        evaluator
                    )
                    action_scores[cand_tuple] += score
                    action_counts[cand_tuple] += 1
                    stats.incr("candidates_scored")
                except Exception:
                    stats.incr("rollout_fail")
        finally:
            try:
                search_end()
            except Exception:
                pass

        determinizations_run += 1

    # Select candidate with highest average score
    best_cand = candidates[0]
    best_avg_score = -float("inf")
    avgs = []

    for cand in candidates:
        ctup = tuple(cand)
        cnt = action_counts[ctup]
        avg = (action_scores[ctup] / cnt) if cnt > 0 else -1e9
        avgs.append(avg)
        if avg > best_avg_score:
            best_avg_score = avg
            best_cand = cand

    if len(avgs) > 1 and max(avgs) - min(avgs) < 1e-9:
        stats.incr("search_degenerate")
    if best_cand != candidates[0]:
        stats.incr("search_picked_nonzero")

    return best_cand
