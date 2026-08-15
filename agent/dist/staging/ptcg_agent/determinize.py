"""Determinization sampling and belief state tracking for PIMC search."""
import random
import sys
from collections import Counter
from pathlib import Path

try:
    from cg.api import Observation, State, CardType, LogType
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import Observation, State, CardType, LogType

from .carddb import get_card_db


def _pokemon_card_ids(p) -> list[int]:
    ids = [p.id]
    for c in p.energyCards or []:
        ids.append(c.id)
    for c in p.tools or []:
        ids.append(c.id)
    for c in p.preEvolution or []:
        ids.append(c.id)
    return ids


def player_visible_ids(player) -> list[int]:
    """All card ids visibly owned by a player: board, discard, face-up prizes."""
    ids: list[int] = []
    if player.active and player.active[0]:
        ids.extend(_pokemon_card_ids(player.active[0]))
    for p in player.bench or []:
        if p:
            ids.extend(_pokemon_card_ids(p))
    for c in player.discard or []:
        ids.append(c.id)
    for c in player.prize or []:
        if c is not None:
            ids.append(c.id)
    return ids


class StateTracker:
    """Exact set-subtraction bookkeeping for own deck and prizes."""

    def __init__(self, full_deck_ids: list[int]):
        self.full_deck_ids = list(full_deck_ids)
        counts = Counter(full_deck_ids)
        self._pad_id = counts.most_common(1)[0][0] if counts else 3

    def get_own_hidden_cards(self, obs: Observation) -> tuple[list[int], list[int]]:
        """Calculate own remaining unrevealed deck and prize cards via set subtraction.

        Returns:
            (your_deck_ids, your_prize_ids)
        """
        if not obs.current:
            return list(self.full_deck_ids), []

        your_idx = obs.current.yourIndex
        me = obs.current.players[your_idx]

        visible_ids: list[int] = list(player_visible_ids(me))
        if me.hand:
            for c in me.hand:
                visible_ids.append(c.id)

        known_prize_ids: list[int] = [c.id for c in me.prize if c is not None]

        remaining_pool = list(self.full_deck_ids)
        for vid in visible_ids:
            if vid in remaining_pool:
                remaining_pool.remove(vid)

        deck_count = me.deckCount
        prize_count = len(me.prize) - len(known_prize_ids)

        random.shuffle(remaining_pool)
        your_deck = remaining_pool[:deck_count]
        your_prize = known_prize_ids + remaining_pool[deck_count : deck_count + prize_count]

        # Pad with our deck's most common card if bookkeeping missed anything.
        while len(your_deck) < me.deckCount:
            your_deck.append(self._pad_id)
        while len(your_prize) < len(me.prize):
            your_prize.append(self._pad_id)

        return your_deck, your_prize


class OpponentBelief:
    """Belief state tracker for opponent cards and archetype sampling."""

    def __init__(self):
        # serial -> cardId; a physical card is re-logged on every move, so
        # dedup by serial or the belief pool inflates without bound.
        self.observed_by_serial: dict[int, int] = {}

    @property
    def observed_opponent_ids(self) -> list[int]:
        return list(self.observed_by_serial.values())

    def update_from_logs(self, obs: Observation):
        if not obs.current or not obs.logs:
            return

        opp_idx = 1 - obs.current.yourIndex
        db = get_card_db()

        for log in obs.logs:
            if getattr(log, "playerIndex", None) != opp_idx:
                continue
            cid = getattr(log, "cardId", None)
            serial = getattr(log, "serial", None)
            if cid and cid > 0 and serial is not None and cid in db.card_by_id:
                self.observed_by_serial[serial] = cid

    def sample_opponent(
        self, obs: Observation, archetype_prior: list[int] | None = None
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Sample plausible opponent (deck, prize, hand, active).

        The hidden-card pool is the archetype prior (default: mirror of our own
        deck, the dominant meta assumption) minus the opponent's visible cards,
        so sampled worlds respect copy counts.
        """
        db = get_card_db()
        opp_idx = 1 - obs.current.yourIndex
        opp = obs.current.players[opp_idx]

        deck_count = opp.deckCount
        prize_count = sum(1 for c in opp.prize if c is None)
        known_prize_ids = [c.id for c in opp.prize if c is not None]
        hand_count = opp.handCount

        opponent_active: list[int] = []

        prior = list(archetype_prior) if archetype_prior else []
        if not prior:
            prior = [3] * 35 + [721] * 2 + [722] * 4 + [723] * 4 + [1145] * 4 \
                + [1158] + [1205] * 2 + [1227] * 4 + [1235] * 4

        pool = Counter(prior)
        for vid in player_visible_ids(opp):
            if pool[vid] > 0:
                pool[vid] -= 1

        prior_basics = [
            cid for cid in set(prior)
            if cid in db.card_by_id
            and db.card_by_id[cid].cardType == CardType.POKEMON
            and db.card_by_id[cid].basic
        ]

        if opp.active and len(opp.active) > 0 and opp.active[0] is None:
            # Face-down active must be a Basic Pokemon.
            candidates = [c for c in prior_basics if pool[c] > 0] or prior_basics or [722]
            chosen = random.choice(candidates)
            opponent_active = [chosen]
            if pool[chosen] > 0:
                pool[chosen] -= 1

        candidate_pool = [cid for cid, cnt in pool.items() for _ in range(cnt) if cnt > 0]

        counts = Counter(prior)
        pad_id = counts.most_common(1)[0][0] if counts else 3
        need = hand_count + prize_count + deck_count
        while len(candidate_pool) < need:
            candidate_pool.append(pad_id)

        random.shuffle(candidate_pool)

        opp_hand = candidate_pool[:hand_count]
        candidate_pool = candidate_pool[hand_count:]

        opp_prize = known_prize_ids + candidate_pool[:prize_count]
        candidate_pool = candidate_pool[prize_count:]

        opp_deck = candidate_pool[:deck_count]

        has_basic = any(
            cid in db.card_by_id and db.card_by_id[cid].cardType == CardType.POKEMON and db.card_by_id[cid].basic
            for cid in opp_deck
        )
        if not has_basic and opp_deck:
            opp_deck[0] = prior_basics[0] if prior_basics else 722

        return opp_deck, opp_prize, opp_hand, opponent_active


def sample_determinization(
    obs: Observation,
    state_tracker: StateTracker,
    belief: OpponentBelief,
) -> dict[str, list[int]]:
    """Sample a complete, legal determinization dict for search_begin."""
    your_deck, your_prize = state_tracker.get_own_hidden_cards(obs)
    opp_deck, opp_prize, opp_hand, opp_active = belief.sample_opponent(
        obs, archetype_prior=state_tracker.full_deck_ids
    )

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opp_deck,
        "opponent_prize": opp_prize,
        "opponent_hand": opp_hand,
        "opponent_active": opp_active,
    }
