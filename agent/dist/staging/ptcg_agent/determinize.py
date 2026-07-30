"""Determinization sampling and belief state tracking for PIMC search."""
import random
import sys
from pathlib import Path

try:
    from cg.api import Observation, State, CardType, LogType
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import Observation, State, CardType, LogType

from .carddb import get_card_db


class StateTracker:
    """Exact set-subtraction bookkeeping for own deck and prizes."""

    def __init__(self, full_deck_ids: list[int]):
        self.full_deck_ids = list(full_deck_ids)

    def get_own_hidden_cards(self, obs: Observation) -> tuple[list[int], list[int]]:
        """Calculate own remaining unrevealed deck and prize cards via set subtraction.
        
        Returns:
            (your_deck_ids, your_prize_ids)
        """
        db = get_card_db()
        if not obs.current:
            return list(self.full_deck_ids), []

        your_idx = obs.current.yourIndex
        me = obs.current.players[your_idx]

        # Account for all visible cards owned by me
        visible_ids: list[int] = []

        # Hand
        if me.hand:
            for c in me.hand:
                visible_ids.append(c.id)

        # Active
        if me.active and me.active[0]:
            p = me.active[0]
            visible_ids.append(p.id)
            for c in p.energyCards:
                visible_ids.append(c.id)
            for c in p.tools:
                visible_ids.append(c.id)
            for c in p.preEvolution:
                visible_ids.append(c.id)

        # Bench
        for p in me.bench:
            visible_ids.append(p.id)
            for c in p.energyCards:
                visible_ids.append(c.id)
            for c in p.tools:
                visible_ids.append(c.id)
            for c in p.preEvolution:
                visible_ids.append(c.id)

        # Discard
        for c in me.discard:
            visible_ids.append(c.id)

        # Known face-up Prize cards
        known_prize_ids: list[int] = []
        for c in me.prize:
            if c is not None:
                known_prize_ids.append(c.id)
                visible_ids.append(c.id)

        # Remaining pool = full_deck_ids - visible_ids
        remaining_pool = list(self.full_deck_ids)
        for vid in visible_ids:
            if vid in remaining_pool:
                remaining_pool.remove(vid)

        # Split remaining pool into deck and hidden prize cards
        deck_count = me.deckCount
        prize_count = len(me.prize) - len(known_prize_ids)

        random.shuffle(remaining_pool)
        your_deck = remaining_pool[:deck_count]
        your_prize = known_prize_ids + remaining_pool[deck_count : deck_count + prize_count]

        # Fill with fallback basic energy if missing due to edge cases
        while len(your_deck) < me.deckCount:
            your_deck.append(2)  # Basic Fire Energy
        while len(your_prize) < len(me.prize):
            your_prize.append(2)

        return your_deck, your_prize


class OpponentBelief:
    """Belief state tracker for opponent cards and archetype sampling."""

    def __init__(self):
        self.observed_opponent_ids: list[int] = []

    def update_from_logs(self, obs: Observation):
        if not obs.current or not obs.logs:
            return

        opp_idx = 1 - obs.current.yourIndex
        db = get_card_db()

        for log in obs.logs:
            if getattr(log, "playerIndex", None) == opp_idx:
                cid = getattr(log, "cardId", None)
                if cid and cid > 0 and cid in db.card_by_id:
                    self.observed_opponent_ids.append(cid)

    def sample_opponent(
        self, obs: Observation
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Sample plausible opponent (deck, prize, hand, active).
        
        Returns:
            (opponent_deck, opponent_prize, opponent_hand, opponent_active)
        """
        db = get_card_db()
        opp_idx = 1 - obs.current.yourIndex
        opp = obs.current.players[opp_idx]

        deck_count = opp.deckCount
        prize_count = len(opp.prize)
        hand_count = opp.handCount

        # Active check (face-down active requires prediction)
        opponent_active: list[int] = []
        if opp.active and len(opp.active) > 0 and opp.active[0] is None:
            # Face-down active -> sample a basic Pokemon ID
            basic_pokes = list(db.basic_pokemon_ids)
            opponent_active = [random.choice(basic_pokes) if basic_pokes else 319]

        # Build candidate pool for opponent deck based on observed cards or meta defaults
        candidate_pool: list[int] = list(self.observed_opponent_ids)

        # Fill candidate pool to at least 60 cards with standard competitive cards
        default_filler = [
            319, 319, 319, 319,  # Charcadet
            797, 797, 797, 797,  # Ceruledge
            1227, 1227, 1227, 1227,  # Lillie's Determination
            1213, 1213, 1213, 1213,  # Judge
            1182, 1182, 1182, 1182,  # Boss's Orders
            1086, 1086, 1086, 1086,  # Buddy-Buddy Poffin
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2  # Basic Energy
        ]
        candidate_pool.extend(default_filler)
        random.shuffle(candidate_pool)

        # Assign to hand, prize, deck
        opp_hand = candidate_pool[:hand_count]
        candidate_pool = candidate_pool[hand_count:]

        opp_prize = candidate_pool[:prize_count]
        candidate_pool = candidate_pool[prize_count:]

        opp_deck = candidate_pool[:deck_count]

        # Ensure deck has at least 1 Basic Pokemon if setup
        has_basic = any(
            cid in db.card_by_id and db.card_by_id[cid].cardType == CardType.POKEMON and db.card_by_id[cid].basic
            for cid in opp_deck
        )
        if not has_basic and opp_deck:
            opp_deck[0] = 319  # Charcadet basic

        # Fill size gaps if needed
        while len(opp_hand) < hand_count:
            opp_hand.append(2)
        while len(opp_prize) < prize_count:
            opp_prize.append(2)
        while len(opp_deck) < deck_count:
            opp_deck.append(2)

        return opp_deck, opp_prize, opp_hand, opponent_active


def sample_determinization(
    obs: Observation,
    state_tracker: StateTracker,
    belief: OpponentBelief,
) -> dict[str, list[int]]:
    """Sample a complete, legal determinization dict for search_begin."""
    your_deck, your_prize = state_tracker.get_own_hidden_cards(obs)
    opp_deck, opp_prize, opp_hand, opp_active = belief.sample_opponent(obs)

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opp_deck,
        "opponent_prize": opp_prize,
        "opponent_hand": opp_hand,
        "opponent_active": opp_active,
    }
