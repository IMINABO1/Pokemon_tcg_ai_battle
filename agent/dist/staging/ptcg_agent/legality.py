"""Deck legality checker.

Mirrors the engine's deck rules (Api.h `ApiBattleStart` / Core.h):
- exactly 60 cards
- max 4 copies per named card, EXCEPT Basic Energy (unlimited)
- at most 1 ACE SPEC card
- at least 1 Basic Pokemon

This is a pre-flight gate so a bad deck never reaches `battle_start` / `search_begin`.
The compiled engine remains the final authority; this just catches the common cases
early with a human-readable reason.
"""
from __future__ import annotations

from collections import Counter

from . import carddb


class DeckLegalityError(ValueError):
    """Raised when a deck violates the construction rules."""


def check_deck(deck: list[int]) -> None:
    """Raise DeckLegalityError if `deck` is illegal; return None if legal."""
    reasons = deck_violations(deck)
    if reasons:
        raise DeckLegalityError("; ".join(reasons))


def is_legal(deck: list[int]) -> bool:
    return not deck_violations(deck)


def deck_violations(deck: list[int]) -> list[str]:
    """Return a list of human-readable rule violations (empty == legal)."""
    reasons: list[str] = []

    if len(deck) != 60:
        reasons.append(f"deck must contain exactly 60 cards (got {len(deck)})")

    unknown = sorted({cid for cid in deck if cid not in carddb.CARD_BY_ID})
    if unknown:
        reasons.append(f"unknown card ids: {unknown}")
        # Can't reason further about unknown cards.
        return reasons

    # Max 4 per *name*, except basic energy (unlimited). Group ids by name so
    # different prints of the same card share the 4-copy limit.
    name_counts: Counter[str] = Counter()
    for cid in deck:
        if carddb.is_basic_energy(cid):
            continue
        name_counts[carddb.CARD_BY_ID[cid].name] += 1
    for name, n in name_counts.items():
        if n > 4:
            reasons.append(f"more than 4 copies of '{name}' ({n})")

    ace_specs = [cid for cid in deck if cid in carddb.ACE_SPEC_IDS]
    if len(ace_specs) > 1:
        names = sorted({carddb.CARD_BY_ID[c].name for c in ace_specs})
        reasons.append(f"more than 1 ACE SPEC card ({names})")

    if not any(carddb.is_basic_pokemon(cid) for cid in deck):
        reasons.append("deck has no Basic Pokemon")

    return reasons
