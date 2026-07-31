"""Unit tests for determinize.py."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from collections import Counter

from ptcg_agent.determinize import StateTracker, OpponentBelief, sample_determinization
from cg.api import Observation, State, PlayerState, Card, Pokemon


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


def _card(cid):
    return Card(id=cid, serial=cid, playerIndex=0)


def _pokemon(cid, energy_ids=()):
    return Pokemon(
        id=cid, serial=cid, hp=100, maxHp=100, appearThisTurn=False,
        energies=[], energyCards=[_card(e) for e in energy_ids], tools=[], preEvolution=[],
    )


def test_own_bookkeeping_conserves_multiset():
    """The key invariant: your_deck + your_prize == full deck minus everything visible.
    A bug here would hand search_begin a determinization inconsistent with reality."""
    # A real 60-card deck: the 2 Fire energies that end up attached are part of it.
    full = [319] * 4 + [797] * 4 + [2] * 2 + list(range(100, 150))  # 60 cards
    assert len(full) == 60

    hand = [_card(100), _card(101), _card(319)]           # 3 in hand
    active = [_pokemon(797, energy_ids=[2, 2])]           # Ceruledge + 2 Fire attached
    discard = [_card(102)]

    me = PlayerState(
        active=active, bench=[], benchMax=5, deckCount=47, discard=discard,
        prize=[None] * 6, handCount=3, hand=hand,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )
    opp = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=50, discard=[], prize=[None] * 6,
        handCount=3, hand=None, poisoned=False, burned=False, asleep=False,
        paralyzed=False, confused=False,
    )
    state = State(
        turn=3, turnActionCount=0, yourIndex=0, firstPlayer=0, supporterPlayed=False,
        stadiumPlayed=False, energyAttached=False, retreated=False, result=-1,
        stadium=[], looking=None, players=[me, opp],
    )
    obs = Observation(select=None, logs=[], current=state)

    tracker = StateTracker(full)
    deck, prize = tracker.get_own_hidden_cards(obs)

    visible = Counter([100, 101, 319, 797, 2, 2, 102])
    assert Counter(deck + prize) == Counter(full) - visible
    assert len(deck) == 47
