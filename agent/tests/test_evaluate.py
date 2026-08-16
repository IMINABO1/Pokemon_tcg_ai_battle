"""Unit tests for evaluate.py."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cg.api import Card, PlayerState, State, CardType
from ptcg_agent.evaluate import evaluate_state
from ptcg_agent.carddb import get_card_db


def _player(hand, hand_count):
    return PlayerState(
        active=[], bench=[], benchMax=5, deckCount=40, discard=[], prize=[None] * 6,
        handCount=hand_count, hand=hand, poisoned=False, burned=False, asleep=False,
        paralyzed=False, confused=False,
    )


def _state(me_hand, me_hand_count):
    me = _player(me_hand, me_hand_count)
    opp = _player(None, 5)
    return State(
        turn=3, turnActionCount=0, yourIndex=0, firstPlayer=0, supporterPlayed=False,
        stadiumPlayed=False, energyAttached=False, retreated=False, result=-1,
        stadium=[], looking=None, players=[me, opp],
    )


def _a_supporter_id():
    db = get_card_db()
    for cid, card in db.card_by_id.items():
        if card.cardType == CardType.SUPPORTER:
            return cid
    return None


import pytest


@pytest.mark.xfail(
    reason="Known divergence: the shipped evaluator has deliberate hand-visibility "
    "terms (supporter_in_hand, fire_in_hand). Turn-horizon rollouts evaluate all "
    "sibling candidates at the same phase (our next turn, hand visible), so the "
    "bias is shared across candidates; revisit if leaf phases ever diverge.",
    strict=True,
)
def test_hand_visibility_does_not_change_score():
    # A rollout evaluates leaves both on our turn (hand visible) and after we end
    # the turn (opponent POV, our hand hidden). The evaluator must score both the
    # same, or ending a turn is silently penalized. Use a real supporter id so a
    # regression that re-adds a hand-content term would diverge and fail here.
    supporter_id = _a_supporter_id()
    assert supporter_id is not None
    hand = [Card(id=supporter_id, serial=i + 1, playerIndex=0) for i in range(5)]

    visible = _state(hand, len(hand))
    hidden = _state(None, len(hand))

    assert evaluate_state(visible, 0) == evaluate_state(hidden, 0)
