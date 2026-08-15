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

from .config import (
    NUM_DETERMINIZATIONS, MAX_ACTION_CANDIDATES,
    ROLLOUT_TURN_HORIZON, MAX_ROLLOUT_DECISIONS,
)
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


def _ranked_option_indices(obs: Observation, n_options: int) -> list[int]:
    """All option indices ordered by descending heuristic card priority
    (stable on ties; unresolvable cards rank last in original order)."""
    scored: list[tuple[int, int]] = []
    for i in range(n_options):
        cid = _option_card_id(obs.select.option[i], obs)
        if cid and cid in carddb.CARD_BY_ID:
            scored.append((_card_priority(carddb.CARD_BY_ID[cid]), i))
        else:
            scored.append((0, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in scored]


def _shortlist_single_options(obs: Observation, n_options: int, k: int) -> list[int]:
    """Pick k option indices for a 1-of-n select, ranked heuristically when the
    option cards are resolvable, rather than blindly taking the first k."""
    if n_options <= k:
        return list(range(n_options))
    return sorted(_ranked_option_indices(obs, n_options)[:k])


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

    # Multi-card selects (discard costs, deck searches, bench setup): offer the
    # heuristic top-k, the first-k, and a couple of shifted windows so search
    # actually has alternatives to compare instead of a single forced pick.
    def _k_subsets(k: int) -> list[list[int]]:
        if n_options <= k:
            return [list(range(n_options))]
        ranked = _ranked_option_indices(obs, n_options)
        subsets = [sorted(ranked[:k]), list(range(k)), sorted(ranked[-k:]),
                   list(range(n_options - k, n_options))]
        seen, out = set(), []
        for s in subsets:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                out.append(s)
        return out

    if min_c == max_c:
        return _k_subsets(max_c)

    limit_c = min(max_c, n_options)
    if limit_c == 0:
        return [[]]
    candidates = []
    if min_c == 0:
        candidates.append([])
    lo = max(min_c, 1)
    for k in {lo, limit_c}:
        candidates.extend(_k_subsets(k))
    seen, out = set(), []
    for s in candidates:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            out.append(s)
    return out[:MAX_ACTION_CANDIDATES]


# Aggressive static rollout policy: applies to BOTH players inside rollouts.
# Development actions outrank ATTACK because attacking ends the turn — the
# rollout should play out a full turn and then attack, like a real player.
_ROLLOUT_PRIORITY = {
    OptionType.EVOLVE: 80,
    OptionType.ATTACH: 75,
    OptionType.ABILITY: 70,
    OptionType.PLAY: 65,
    OptionType.ATTACK: 60,
    OptionType.YES: 40,
    OptionType.NUMBER: 30,
    OptionType.CARD: 20,
    OptionType.TOOL_CARD: 20,
    OptionType.ENERGY_CARD: 20,
    OptionType.ENERGY: 20,
    OptionType.SKILL: 20,
    OptionType.SPECIAL_CONDITION: 20,
    OptionType.NO: 10,
    OptionType.END: 5,
    OptionType.RETREAT: 0,
}


def _rollout_policy_action(obs: Observation) -> list[int]:
    """Single fast action for rollout continuation — no branching, no eval."""
    select = obs.select
    options = select.option
    n_options = len(options)
    min_c = select.minCount
    max_c = min(select.maxCount, n_options)

    if n_options == 0 or max_c == 0:
        return []

    def prio(opt) -> float:
        p = _ROLLOUT_PRIORITY.get(opt.type, 20)
        if opt.type == OptionType.NUMBER and opt.number is not None:
            p += min(opt.number, 9) * 0.1
        return p

    if min_c <= 1:
        best_i = max(range(n_options), key=lambda i: prio(options[i]))
        if min_c == 0 and prio(options[best_i]) < 20:
            return []
        return [best_i]

    count = min(min_c, max_c)
    ranked = sorted(range(n_options), key=lambda i: -prio(options[i]))
    return sorted(ranked[:count])


def _rollout(
    start_search_state: SearchState,
    initial_action: list[int],
    your_index: int,
    evaluator: Callable[[State, int], float],
    budget: BudgetTracker | None = None,
    root_turn: int = 0,
) -> float:
    """Apply initial_action, then continue with the static rollout policy until
    the start of our turn after next (root_turn + ROLLOUT_TURN_HORIZON), so every
    candidate line is evaluated at the same game phase. Fixed-decision-depth
    rollouts made turn-ending actions (attacks) look worse than stalling,
    because only attack lines ever showed the opponent's reply."""
    current_search_state = search_step(start_search_state.searchId, initial_action)

    steps = 0
    while steps < MAX_ROLLOUT_DECISIONS:
        obs = current_search_state.observation
        if not obs or not obs.select or not obs.current:
            break
        if obs.current.result != -1:
            break
        if obs.current.turn >= root_turn + ROLLOUT_TURN_HORIZON:
            break
        if budget is not None and budget.is_expired():
            break
        action = _rollout_policy_action(obs)
        try:
            current_search_state = search_step(current_search_state.searchId, action)
        except Exception:
            break
        steps += 1

    leaf_obs = current_search_state.observation
    if leaf_obs and leaf_obs.current:
        return evaluator(leaf_obs.current, your_index)
    return 0.0


def _legal_determinization(obs, state_tracker, belief, retries: int):
    """Sample a determinization whose opponent hidden zones pass the copy-count
    gate, with bounded retries; return the last sample regardless so search
    never stalls. The check must include the opponent's VISIBLE cards and must
    not require 60 cards — a full-deck check can never pass once the opponent
    has cards in play."""
    from .legality import partial_ok
    from .determinize import player_visible_ids

    visible = []
    if obs.current:
        visible = player_visible_ids(obs.current.players[1 - obs.current.yourIndex])

    det = None
    for _ in range(max(1, retries)):
        det = sample_determinization(obs, state_tracker, belief)
        hidden = (det["opponent_deck"] + det["opponent_prize"]
                  + det["opponent_hand"] + det["opponent_active"])
        if partial_ok(hidden + visible):
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
    root_turn = obs.current.turn if obs.current else 0
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
                        your_index,
                        evaluator,
                        budget,
                        root_turn
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
