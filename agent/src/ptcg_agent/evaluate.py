"""Heuristic evaluator for PTCG game states."""
import sys
from pathlib import Path

try:
    from cg.api import State, PlayerState, Pokemon, CardType
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import State, PlayerState, Pokemon, CardType

from .config import EVAL_WEIGHTS
from .carddb import get_card_db, energy_cost_met, compute_damage


def _score_prizes(me: PlayerState, opp: PlayerState) -> float:
    # Standard PTCG has 6 prize cards initially.
    # Prizes taken = 6 - len(prizes)
    my_prizes_remaining = len(me.prize)
    opp_prizes_remaining = len(opp.prize)

    my_prizes_taken = 6 - my_prizes_remaining
    opp_prizes_taken = 6 - opp_prizes_remaining

    prize_diff = my_prizes_taken - opp_prizes_taken
    return (
        prize_diff * EVAL_WEIGHTS["prize_diff"]
        + my_prizes_remaining * EVAL_WEIGHTS["own_prizes_remaining"]
    )


def _hp_fraction_sum(player: PlayerState) -> float:
    total = 0.0
    pokemons = []
    if player.active and player.active[0]:
        pokemons.append(player.active[0])
    pokemons.extend(player.bench)
    for p in pokemons:
        if p.maxHp > 0:
            total += p.hp / p.maxHp
    return total


def _score_hp(me: PlayerState, opp: PlayerState) -> float:
    hp_diff = _hp_fraction_sum(me) - _hp_fraction_sum(opp)
    return hp_diff * EVAL_WEIGHTS["hp_swing"]


def _score_lethal_threat(me: PlayerState, opp: PlayerState) -> float:
    score = 0.0
    db = get_card_db()

    if not me.active or not me.active[0] or not opp.active or not opp.active[0]:
        return score

    my_active = me.active[0]
    opp_active = opp.active[0]

    my_card = db.card_by_id.get(my_active.id)
    opp_card = db.card_by_id.get(opp_active.id)

    if not my_card or not opp_card:
        return score

    # Check if my active can KO opponent's active right now
    for atk_id in my_card.attacks:
        attack = db.attack_by_id.get(atk_id)
        if attack and energy_cost_met(attack, my_active.energies):
            tools = [db.card_by_id[t.id] for t in my_active.tools if t.id in db.card_by_id]
            dmg = compute_damage(my_card, attack, opp_card, tools=tools)
            if dmg >= opp_active.hp:
                score += EVAL_WEIGHTS["can_ko_active"]
                break

    # Check if opponent active threatens my active
    for atk_id in opp_card.attacks:
        attack = db.attack_by_id.get(atk_id)
        if attack and energy_cost_met(attack, opp_active.energies):
            tools = [db.card_by_id[t.id] for t in opp_active.tools if t.id in db.card_by_id]
            dmg = compute_damage(opp_card, attack, my_card, tools=tools)
            if dmg >= my_active.hp:
                score += EVAL_WEIGHTS["active_in_danger"]
                break

    return score


def _score_energy_tempo(me: PlayerState) -> float:
    score = 0.0
    db = get_card_db()

    total_attached = 0
    attack_ready = False

    pokemons: list[Pokemon] = []
    if me.active and me.active[0]:
        pokemons.append(me.active[0])
    pokemons.extend(me.bench)

    for p in pokemons:
        total_attached += len(p.energies)
        card = db.card_by_id.get(p.id)
        if card:
            for atk_id in card.attacks:
                attack = db.attack_by_id.get(atk_id)
                if attack and energy_cost_met(attack, p.energies):
                    attack_ready = True
                    break

    score += total_attached * EVAL_WEIGHTS["energy_attached"]
    if attack_ready:
        score += EVAL_WEIGHTS["attack_ready"]

    return score


def _score_board(me: PlayerState) -> float:
    score = 0.0
    bench_count = len(me.bench)
    score += bench_count * EVAL_WEIGHTS["bench_count"]

    # Evolution advantage
    evolved_count = 0
    if me.active and me.active[0] and me.active[0].preEvolution:
        evolved_count += len(me.active[0].preEvolution)
    for p in me.bench:
        if p.preEvolution:
            evolved_count += len(p.preEvolution)

    score += evolved_count * EVAL_WEIGHTS["evolution_advantage"]
    return score


def _score_hand(me: PlayerState) -> float:
    score = 0.0
    db = get_card_db()

    hand_size = me.handCount if me.hand is None else len(me.hand)
    score += hand_size * EVAL_WEIGHTS["hand_size"]

    if me.hand is not None:
        has_supporter = any(
            db.card_by_id.get(c.id) and db.card_by_id[c.id].cardType == CardType.SUPPORTER
            for c in me.hand
        )
        if has_supporter:
            score += EVAL_WEIGHTS["supporter_in_hand"]

    return score


def _score_status(me: PlayerState) -> float:
    score = 0.0
    if me.poisoned:
        score += EVAL_WEIGHTS["status_poisoned"]
    if me.burned:
        score += EVAL_WEIGHTS["status_burned"]
    if me.asleep:
        score += EVAL_WEIGHTS["status_asleep"]
    if me.paralyzed:
        score += EVAL_WEIGHTS["status_paralyzed"]
    if me.confused:
        score += EVAL_WEIGHTS["status_confused"]
    return score


def evaluate_state(state: State, your_index: int) -> float:
    """Evaluate a game state from player your_index's perspective."""
    if state is None:
        return 0.0

    # Terminal state checks
    if state.result != -1:
        if state.result == your_index:
            return 1_000_000.0
        elif state.result == (1 - your_index):
            return -1_000_000.0
        else:
            return 0.0  # Draw

    me = state.players[your_index]
    opp = state.players[1 - your_index]

    total_score = (
        _score_prizes(me, opp)
        + _score_hp(me, opp)
        + _score_lethal_threat(me, opp)
        + _score_energy_tempo(me)
        + _score_board(me)
        + _score_hand(me)
        + _score_status(me)
    )

    return total_score
