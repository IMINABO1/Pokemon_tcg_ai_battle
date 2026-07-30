"""Card knowledge base and deck legality validation."""
import sys
from pathlib import Path
from typing import Callable, Optional, Union

# Ensure sample_submission cg is available if not already in sys.path
try:
    from cg.api import (
        all_card_data, all_attack, CardData, Attack, CardType, EnergyType
    )
except ImportError:
    possible_cg = Path(__file__).resolve().parent.parent.parent.parent / "sample_submission" / "sample_submission"
    if possible_cg.exists() and str(possible_cg) not in sys.path:
        sys.path.insert(0, str(possible_cg))
    from cg.api import (
        all_card_data, all_attack, CardData, Attack, CardType, EnergyType
    )


class CardDatabase:
    """Singleton-style card database populated from cg.api."""
    _instance: Optional["CardDatabase"] = None

    def __init__(self):
        self.cards: list[CardData] = all_card_data()
        self.attacks: list[Attack] = all_attack()

        self.card_by_id: dict[int, CardData] = {c.cardId: c for c in self.cards}
        self.attack_by_id: dict[int, Attack] = {a.attackId: a for a in self.attacks}

        self.cards_by_name: dict[str, list[CardData]] = {}
        for c in self.cards:
            self.cards_by_name.setdefault(c.name, []).append(c)

        self.basic_energy_ids: set[int] = {
            c.cardId for c in self.cards if c.cardType == CardType.BASIC_ENERGY
        }
        self.ace_spec_ids: set[int] = {c.cardId for c in self.cards if c.aceSpec}
        self.basic_pokemon_ids: set[int] = {
            c.cardId for c in self.cards if c.cardType == CardType.POKEMON and c.basic
        }

    @classmethod
    def get_instance(cls) -> "CardDatabase":
        if cls._instance is None:
            cls._instance = CardDatabase()
        return cls._instance


# Initialize singleton database instance
_DB = CardDatabase.get_instance()

# Module-level alias exports for backward compatibility & direct access
CARD_BY_ID = _DB.card_by_id
ATTACK_BY_ID = _DB.attack_by_id
CARDS_BY_NAME = _DB.cards_by_name
BASIC_ENERGY_IDS = _DB.basic_energy_ids
ACE_SPEC_IDS = _DB.ace_spec_ids
BASIC_POKEMON_IDS = _DB.basic_pokemon_ids


def get_card_db() -> CardDatabase:
    return CardDatabase.get_instance()


def is_basic_energy(cid: int) -> bool:
    return cid in BASIC_ENERGY_IDS


def is_basic_pokemon(cid: int) -> bool:
    return cid in BASIC_POKEMON_IDS


def energy_cost_met(attack: Attack, attached: list[EnergyType]) -> bool:
    """Check if attached energies satisfy the attack cost requirements.
    
    Colorless energy costs can be satisfied by any energy type.
    Rainbow energy can satisfy any specific typed requirement or colorless.
    """
    if not attack.energies:
        return True
    
    attached_counts = {}
    rainbow_count = 0

    for e in attached:
        e_val = int(e)
        if e_val == int(EnergyType.RAINBOW):
            rainbow_count += 1
        else:
            attached_counts[e_val] = attached_counts.get(e_val, 0) + 1

    colorless_req = 0
    for req in attack.energies:
        req_val = int(req)
        if req_val == int(EnergyType.COLORLESS):
            colorless_req += 1
        else:
            if attached_counts.get(req_val, 0) > 0:
                attached_counts[req_val] -= 1
            elif rainbow_count > 0:
                rainbow_count -= 1
            else:
                return False
    
    remaining_attached = sum(attached_counts.values()) + rainbow_count
    return remaining_attached >= colorless_req


def compute_damage(
    arg1: Union[CardData, Attack],
    arg2: CardData,
    defender: Optional[CardData] = None,
    tools: Optional[list[CardData]] = None,
    boosts: int = 0
) -> int:
    """Damage estimation function supporting both:
    - compute_damage(attack, defender)
    - compute_damage(attacker, attack, defender, tools, boosts)
    """
    if isinstance(arg1, Attack):
        attack = arg1
        def_card = arg2
        attacker_card = None
    else:
        attacker_card = arg1
        attack = arg2  # type: ignore
        def_card = defender  # type: ignore

    base_dmg = attack.damage + boosts

    if tools and def_card:
        for tool in tools:
            if tool.name == "Maximum Belt" and getattr(def_card, "ex", False):
                base_dmg += 50

    if base_dmg <= 0 or def_card is None:
        return max(0, base_dmg)

    # Weakness (x2 in PTCG)
    energy_type = attacker_card.energyType if attacker_card else (attack.energies[0] if attack.energies else None)
    if energy_type and def_card.weakness is not None and int(def_card.weakness) == int(energy_type):
        base_dmg *= 2

    # Resistance (-30 in modern PTCG)
    if energy_type and def_card.resistance is not None and int(def_card.resistance) == int(energy_type):
        base_dmg = max(0, base_dmg - 30)


    return base_dmg


def validate_deck_legality(deck_card_ids: list[int]) -> tuple[bool, str]:
    """Validate a 60-card deck against PTCG rules."""
    from .legality import deck_violations
    reasons = deck_violations(deck_card_ids)
    if reasons:
        return False, "; ".join(reasons)
    return True, "Deck is legal."
