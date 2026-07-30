"""Smoke test for search and policy execution."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ptcg_agent.policy import agent_decide, _fallback_decide
from ptcg_agent.search import enumerate_candidate_actions
from cg.api import Observation, SelectData, SelectType, Option, OptionType


def test_fallback_decide():
    select = SelectData(
        type=SelectType.MAIN,
        context=0,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=[Option(type=OptionType.END), Option(type=OptionType.PLAY)],
        deck=None,
        contextCard=None,
        effect=None
    )
    obs = Observation(select=select, logs=[], current=None)
    choice = _fallback_decide(obs)
    assert choice == [0] or choice == [1]


def test_enumerate_candidate_actions():
    select = SelectData(
        type=SelectType.YES_NO,
        context=0,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=[Option(type=OptionType.YES), Option(type=OptionType.NO)],
        deck=None,
        contextCard=None,
        effect=None
    )
    obs = Observation(select=select, logs=[], current=None)
    cands = enumerate_candidate_actions(obs)
    assert cands == [[0], [1]]


def test_agent_decide_deck_selection():
    # Initial selection prompt: obs.select is None
    obs_dict = {"select": None, "logs": [], "current": None}
    deck = agent_decide(obs_dict)
    assert len(deck) == 60
