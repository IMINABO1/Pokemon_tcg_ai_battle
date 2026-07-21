"""Card knowledge base.

Loads once at import from the compiled engine (`cg.api.all_card_data()` / `all_attack()`),
NOT from the CSVs (those are offline human-research artifacts). Pure / side-effect-free
after load.

Deliberately out of scope for Phase 1: a general attack/ability text parser. The real
search engine (`state.step()` via search_begin/search_step) is the ground truth for what a
move *does*; the heuristic here only needs to score resulting states and cheaply prune
implausible moves.
"""
from __future__ import annotations

from cg.api import (
    all_card_data,
    all_attack,
    CardData,
    Attack,
    CardType,
    EnergyType,
)

# ---------------------------------------------------------------------------
# One-time load
# ---------------------------------------------------------------------------
_ALL_CARDS: list[CardData] = all_card_data()
_ALL_ATTACKS: list[Attack] = all_attack()

CARD_BY_ID: dict[int, CardData] = {c.cardId: c for c in _ALL_CARDS}
ATTACK_BY_ID: dict[int, Attack] = {a.attackId: a for a in _ALL_ATTACKS}

# name -> list of card ids sharing that name (different arts/prints share a name).
CARDS_BY_NAME: dict[str, list[int]] = {}
for _c in _ALL_CARDS:
    CARDS_BY_NAME.setdefault(_c.name, []).append(_c.cardId)

BASIC_ENERGY_IDS: frozenset[int] = frozenset(
    c.cardId for c in _ALL_CARDS if c.cardType == CardType.BASIC_ENERGY
)
ACE_SPEC_IDS: frozenset[int] = frozenset(c.cardId for c in _ALL_CARDS if c.aceSpec)

# Energy types that satisfy *any* colored requirement.
_WILD_ENERGY = frozenset({EnergyType.RAINBOW})


def is_basic_energy(card_id: int) -> bool:
    return card_id in BASIC_ENERGY_IDS


def is_pokemon(card_id: int) -> bool:
    c = CARD_BY_ID.get(card_id)
    return c is not None and c.cardType == CardType.POKEMON


def is_basic_pokemon(card_id: int) -> bool:
    c = CARD_BY_ID.get(card_id)
    return c is not None and c.cardType == CardType.POKEMON and c.basic


# ---------------------------------------------------------------------------
# Energy cost feasibility (cheap filter, NOT authoritative — engine is ground truth)
# ---------------------------------------------------------------------------
def energy_cost_met(attack: Attack, attached: list[EnergyType]) -> bool:
    """True if `attached` energies can plausibly pay `attack.energies`.

    COLORLESS requirements are paid by any energy; RAINBOW attached energy pays any
    requirement. This is a first-order feasibility check for move-generation pruning
    and opponent-hand plausibility — the real engine remains the authority on legality.
    """
    required = list(attack.energies)
    pool = list(attached)

    # Pay the specific colored requirements first (colorless is the flexible remainder).
    colored = [e for e in required if e != EnergyType.COLORLESS]
    colorless_count = len(required) - len(colored)

    for req in colored:
        # exact-type match preferred, else a wild (rainbow) energy.
        if req in pool:
            pool.remove(req)
        else:
            wild = next((e for e in pool if e in _WILD_ENERGY), None)
            if wild is None:
                return False
            pool.remove(wild)

    # Remaining pool pays the colorless portion (any energy counts).
    return len(pool) >= colorless_count


# ---------------------------------------------------------------------------
# First-order damage estimate (base + weakness/resistance only)
# ---------------------------------------------------------------------------
def compute_damage(attack: Attack, defender: CardData) -> int:
    """First-order damage estimate: base damage adjusted for weakness / resistance.

    Explicitly does NOT parse conditional attack text (e.g. "+30 if...", coin flips,
    damage-counter effects). For those the real engine is the source of truth. A small
    hand-curated override table can extend this later for our key cards / common threats.
    """
    dmg = attack.damage
    if dmg <= 0 or defender is None:
        return max(dmg, 0)

    atk_type = _attack_type(attack)
    if defender.weakness is not None and atk_type == defender.weakness:
        dmg *= 2
    if defender.resistance is not None and atk_type == defender.resistance:
        dmg = max(0, dmg - 30)  # standard resistance reduction
    return dmg


def _attack_type(attack: Attack) -> EnergyType | None:
    """Best-effort attacking type: first non-colorless required energy."""
    for e in attack.energies:
        if e != EnergyType.COLORLESS:
            return e
    return None
