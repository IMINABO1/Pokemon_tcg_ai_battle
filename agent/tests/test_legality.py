"""Comprehensive test suite for PTCG deck legality rules."""
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ptcg_agent.carddb import validate_deck_legality, get_card_db


def test_valid_deck():
    # 4x Charcadet (319), 4x Ceruledge (797), 1x Max Belt (1158), 51x Fire Energy (2)
    deck = [319] * 4 + [797] * 4 + [1158] * 1 + [2] * 51
    is_legal, msg = validate_deck_legality(deck)
    assert is_legal, msg


def test_wrong_card_count():
    deck = [319] * 4 + [2] * 50
    is_legal, msg = validate_deck_legality(deck)
    assert not is_legal
    assert "60 cards" in msg.lower()


def test_no_basic_pokemon():
    # 60 Basic Energy
    deck = [2] * 60
    is_legal, msg = validate_deck_legality(deck)
    assert not is_legal
    assert "basic pokemon" in msg.lower()


def test_too_many_ace_specs():
    # 2x Max Belt (1158)
    deck = [319] * 4 + [1158] * 2 + [2] * 54
    is_legal, msg = validate_deck_legality(deck)
    assert not is_legal
    assert "ace spec" in msg.lower()


def test_too_many_same_name_cards():
    # 5x Charcadet (319)
    deck = [319] * 5 + [2] * 55
    is_legal, msg = validate_deck_legality(deck)
    assert not is_legal
    assert "more than 4" in msg.lower()
