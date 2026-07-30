"""Unit tests for determinize.py."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ptcg_agent.determinize import StateTracker, OpponentBelief, sample_determinization
from cg.api import Observation, State, PlayerState


def test_state_tracker():
    deck_ids = [319] * 4 + [797] * 4 + [2] * 52
    tracker = StateTracker(deck_ids)

    # Mock observation with empty/setup state
    p0 = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=53, discard=[], prize=[None]*6,
        handCount=7, hand=None, poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False
    )
    p1 = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=53, discard=[], prize=[None]*6,
        handCount=7, hand=None, poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False
    )
    state = State(
        turn=1, turnActionCount=0, yourIndex=0, firstPlayer=0, supporterPlayed=False,
        stadiumPlayed=False, energyAttached=False, retreated=False, result=-1, stadium=[], looking=None, players=[p0, p1]
    )
    obs = Observation(select=None, logs=[], current=state)

    your_deck, your_prize = tracker.get_own_hidden_cards(obs)
    assert len(your_deck) == 53
    assert len(your_prize) == 6


def test_sample_determinization():
    deck_ids = [319] * 4 + [797] * 4 + [2] * 52
    tracker = StateTracker(deck_ids)
    belief = OpponentBelief()

    p0 = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=47, discard=[], prize=[None]*6,
        handCount=7, hand=None, poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False
    )
    p1 = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=47, discard=[], prize=[None]*6,
        handCount=7, hand=None, poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False
    )
    state = State(
        turn=1, turnActionCount=0, yourIndex=0, firstPlayer=0, supporterPlayed=False,
        stadiumPlayed=False, energyAttached=False, retreated=False, result=-1, stadium=[], looking=None, players=[p0, p1]
    )
    obs = Observation(select=None, logs=[], current=state)

    det = sample_determinization(obs, tracker, belief)
    assert len(det["your_deck"]) == 47
    assert len(det["your_prize"]) == 6
    assert len(det["opponent_deck"]) == 47
    assert len(det["opponent_prize"]) == 6
    assert len(det["opponent_hand"]) == 7
