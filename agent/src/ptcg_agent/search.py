"""PIMC Search Engine with shallow expectimax rollout and root averaging."""
import sys
from pathlib import Path
from typing import Callable

try:
    from cg.api import (
        Observation, SearchState, search_begin, search_step, search_end,
        SelectType, OptionType, State
    )
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import (
        Observation, SearchState, search_begin, search_step, search_end,
        SelectType, OptionType, State
    )

from .config import NUM_DETERMINIZATIONS, MAX_SEARCH_DEPTH, MAX_ACTION_CANDIDATES
from .budget import BudgetTracker
from .evaluate import evaluate_state
from .determinize import StateTracker, OpponentBelief, sample_determinization
from . import carddb


def _card_priority(card) -> int:
    """Cheap ranking for shortlisting large CARD selects: Pokemon > trainers > energy."""
    ct = card.cardType
    if ct == carddb.CardType.POKEMON:
        return 3
    if ct in (carddb.CardType.SUPPORTER, carddb.CardType.ITEM, carddb.CardType.TOOL):
        return 2
    return 1  # energy / stadium / other


def _option_card_id(opt, obs: Observation) -> int:
    """Best-effort resolution of the card id behind a CARD option, or 0 if unknown."""
    try:
        from cg.api import AreaType
        if opt.type != OptionType.CARD or opt.index is None:
            return 0
        select = obs.select
        if select.deck and 0 <= opt.index < len(select.deck):
            c = select.deck[opt.index]
            return c.id if c is not None else 0
        state = obs.current
        if state is None or opt.area is None:
            return 0
        pidx = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
        player = state.players[pidx]
        cards = None
        if opt.area == AreaType.HAND:
            cards = player.hand
        elif opt.area == AreaType.DISCARD:
            cards = player.discard
        if cards and 0 <= opt.index < len(cards):
            c = cards[opt.index]
            return c.id if c is not None else 0
    except Exception:
        return 0
    return 0


def _shortlist_single_options(obs: Observation, n_options: int, k: int) -> list[int]:
    """Pick k option indices for a 1-of-n select, ranked heuristically when the
    option cards are resolvable, rather than blindly taking the first k."""
    if n_options <= k:
        return list(range(n_options))
    scored: list[tuple[int, int]] = []
    resolvable = False
    for i in range(n_options):
        cid = _option_card_id(obs.select.option[i], obs)
        if cid and cid in carddb.CARD_BY_ID:
            resolvable = True
            scored.append((_card_priority(carddb.CARD_BY_ID[cid]), i))
        else:
            scored.append((0, i))
    if not resolvable:
        return list(range(k))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return sorted(idx for _, idx in scored[:k])


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
        return [[i] for i in _shortlist_single_options(obs, n_options, MAX_ACTION_CANDIDATES)]

    # 0 or 1 selection (e.g. optional action) — include the "pass" option.
    if min_c == 0 and max_c == 1:
        shortlist = _shortlist_single_options(obs, n_options, MAX_ACTION_CANDIDATES - 1)
        return [[]] + [[i] for i in shortlist]

    # Select exactly maxCount options (e.g. setup active/bench or multi-card select)
    if min_c == max_c:
        if n_options <= max_c:
            return [list(range(n_options))]
        return [list(range(max_c))]

    # Default fallback: pick maxCount elements
    limit_c = min(max_c, n_options)
    if limit_c == 0:
        return [[]]
    return [list(range(limit_c))]


def _greedy_next(search_state: SearchState, our_index: int,
                 evaluator: Callable[[State, int], float]) -> SearchState | None:
    """Advance one decision by the option that best serves the acting player:
    maximise our evaluation on our decisions, minimise it on the opponent's."""
    obs = search_state.observation
    if not obs or not obs.select or (obs.current and obs.current.result != -1):
        return None

    candidates = enumerate_candidate_actions(obs)
    if not candidates:
        return None

    acting_index = obs.current.yourIndex if obs.current else our_index
    maximize = acting_index == our_index

    best_state = None
    best_score = None
    for cand in candidates:
        try:
            nxt = search_step(search_state.searchId, cand)
        except Exception:
            continue
        leaf = nxt.observation
        score = evaluator(leaf.current, our_index) if leaf and leaf.current else 0.0
        if best_score is None or (score > best_score if maximize else score < best_score):
            best_score, best_state = score, nxt
    return best_state


def _rollout(
    start_search_state: SearchState,
    initial_action: list[int],
    depth: int,
    your_index: int,
    evaluator: Callable[[State, int], float]
) -> float:
    """Apply initial_action, then greedy 1-ply continuation to `depth`, score the leaf."""
    current_search_state = search_step(start_search_state.searchId, initial_action)

    current_depth = 1
    while current_depth < depth:
        nxt = _greedy_next(current_search_state, your_index, evaluator)
        if nxt is None:
            break
        current_search_state = nxt
        current_depth += 1

    leaf_obs = current_search_state.observation
    if leaf_obs and leaf_obs.current:
        return evaluator(leaf_obs.current, your_index)
    return 0.0


def _legal_determinization(obs, state_tracker, belief, retries: int):
    """Sample a determinization whose opponent deck passes the legality gate, with
    bounded retries; return the last sample regardless so search never stalls."""
    from .legality import is_legal
    det = None
    for _ in range(max(1, retries)):
        det = sample_determinization(obs, state_tracker, belief)
        if is_legal(det["opponent_deck"] + det["opponent_prize"] + det["opponent_hand"]):
            return det
    return det


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
    from .config import DETERMINIZATION_LEGALITY_RETRIES

    candidates = enumerate_candidate_actions(obs)
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates[0]

    your_index = obs.current.yourIndex if obs.current else 0
    action_scores: dict[tuple[int, ...], float] = {tuple(c): 0.0 for c in candidates}
    action_counts: dict[tuple[int, ...], int] = {tuple(c): 0 for c in candidates}

    determinizations_run = 0

    while determinizations_run < NUM_DETERMINIZATIONS and not budget.is_expired():
        det = _legal_determinization(obs, state_tracker, belief, DETERMINIZATION_LEGALITY_RETRIES)

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
            determinizations_run += 1
            continue

        try:
            for cand in candidates:
                if budget.is_expired():
                    break
                cand_tuple = tuple(cand)
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
                except Exception:
                    pass
        finally:
            try:
                search_end()
            except Exception:
                pass

        determinizations_run += 1

    # Select candidate with highest average score
    best_cand = candidates[0]
    best_avg_score = -float("inf")

    for cand in candidates:
        ctup = tuple(cand)
        cnt = action_counts[ctup]
        avg = (action_scores[ctup] / cnt) if cnt > 0 else -1e9
        if avg > best_avg_score:
            best_avg_score = avg
            best_cand = cand

    return best_cand
