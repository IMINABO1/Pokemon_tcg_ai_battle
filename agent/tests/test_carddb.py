"""Unit tests for carddb.py."""
import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ptcg_agent.carddb import (
    get_card_db, energy_cost_met, compute_damage, validate_deck_legality
)
from cg.api import EnergyType, CardType


def test_card_db_load():
    db = get_card_db()
    assert len(db.cards) == 1267
    assert len(db.attacks) == 1556
    assert 797 in db.card_by_id  # Ceruledge
    assert 2 in db.basic_energy_ids  # Fire Energy


def test_energy_cost_met():
    db = get_card_db()
    ceruledge = db.card_by_id[797]
    infernal_slash_atk = db.attack_by_id[ceruledge.attacks[0]]

    # Requires 1 Fire energy
    assert energy_cost_met(infernal_slash_atk, [EnergyType.FIRE])
    assert energy_cost_met(infernal_slash_atk, [EnergyType.FIRE, EnergyType.WATER])
    assert not energy_cost_met(infernal_slash_atk, [EnergyType.WATER])


def test_compute_damage():
    db = get_card_db()
    ceruledge = db.card_by_id[797]
    infernal_slash_atk = db.attack_by_id[ceruledge.attacks[0]]
    charcadet = db.card_by_id[319]

    # Base damage 220
    dmg = compute_damage(ceruledge, infernal_slash_atk, charcadet)
    assert dmg == 220


def test_deck_legality():
    db = get_card_db()
    # Create legal deck: 4x Charcadet (319), 4x Ceruledge (797), 1x Max Belt (1158), 51x Fire Energy (2)
    deck = [319] * 4 + [797] * 4 + [1158] * 1 + [2] * 51
    is_legal, reason = validate_deck_legality(deck)
    assert is_legal, reason

    # Invalid size
    is_legal, reason = validate_deck_legality(deck[:50])
    assert not is_legal

    # More than 4 non-basic energy copies
    illegal_deck = [319] * 5 + [797] * 4 + [2] * 51
    is_legal, reason = validate_deck_legality(illegal_deck)
    assert not is_legal
