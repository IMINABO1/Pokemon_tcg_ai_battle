"""Unit tests for carddb + legality. Run: pytest (from agent/)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cg.api import Attack, EnergyType  # noqa: E402
from ptcg_agent import carddb, legality  # noqa: E402


def test_carddb_loaded():
    assert len(carddb.CARD_BY_ID) == 1267
    assert carddb.ATTACK_BY_ID
    assert carddb.BASIC_ENERGY_IDS


def test_energy_cost_colorless_paid_by_anything():
    atk = Attack(attackId=-1, name="t", text="", damage=10,
                 energies=[EnergyType.COLORLESS, EnergyType.COLORLESS])
    assert carddb.energy_cost_met(atk, [EnergyType.WATER, EnergyType.FIRE])
    assert not carddb.energy_cost_met(atk, [EnergyType.WATER])


def test_energy_cost_typed_requirement():
    atk = Attack(attackId=-1, name="t", text="", damage=10,
                 energies=[EnergyType.FIRE, EnergyType.COLORLESS])
    assert carddb.energy_cost_met(atk, [EnergyType.FIRE, EnergyType.WATER])
    assert not carddb.energy_cost_met(atk, [EnergyType.WATER, EnergyType.WATER])


def test_rainbow_pays_any():
    atk = Attack(attackId=-1, name="t", text="", damage=10, energies=[EnergyType.FIRE])
    assert carddb.energy_cost_met(atk, [EnergyType.RAINBOW])


def test_compute_damage_weakness():
    # find a defender with a weakness and an attack of that type
    defender = next(c for c in carddb.CARD_BY_ID.values() if c.weakness is not None and c.hp)
    atk = Attack(attackId=-1, name="t", text="", damage=50, energies=[defender.weakness])
    assert carddb.compute_damage(atk, defender) == 100


def test_legality_sample_deck_legal():
    deck_path = os.path.join(os.path.dirname(__file__), "..", "src", "ptcg_agent", "deck.csv")
    deck = [int(x) for x in open(deck_path).read().split("\n")[:60]]
    assert legality.is_legal(deck), legality.deck_violations(deck)


def test_legality_wrong_size():
    assert not legality.is_legal([3] * 59)


def test_legality_too_many_copies():
    # 5 copies of a non-basic-energy card (card 721 is a Pokemon in the sample deck)
    deck = [721] * 5 + [3] * 55
    v = legality.deck_violations(deck)
    assert any("more than 4" in r for r in v)


def test_legality_no_basic_pokemon():
    deck = [3] * 60  # all basic energy, no Pokemon
    v = legality.deck_violations(deck)
    assert any("no Basic Pokemon" in r for r in v)
