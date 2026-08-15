# Deck Candidates & Performance Notes

## Candidate v2 (LIVE): Sample Water Deck (Kyogre / Mega Abomasnow ex)
- **Why**: Under the PIMC searcher this list beat random 29/32 and the greedy
  mirror 32/32; the Ceruledge list managed only ~50%/75% on the same harness.
  Its attacks have no hand-discard riders (nothing to whiff), Mega Abomasnow ex
  has 350 HP, and the ladder is full of sample-deck forks — mirrors are decided
  by play skill, where search wins.
- **List**: 35x Water Energy (#3), 2x Kyogre (#721), 4x Snover (#722),
  4x Mega Abomasnow ex (#723), 4x #1145, 1x Maximum Belt (#1158, ACE SPEC),
  2x Cyrano (#1205), 4x Lillie's Determination (#1227), 4x #1235.
- **Key engine notes**: Hammer-lanche self-mills 6 and scales with Water hits;
  Riptide scales with Water in discard then shuffles it back. The engine prices
  both correctly inside search rollouts — no evaluator special-casing needed.

## Candidate v1 (retired): Ceruledge Fire Efficient Attacker
- Retired 2026-08-15: Infernal Slash ("discard 4 Basic Fire Energy from hand,
  else this attack does nothing") made the deck whiff-prone; even with a
  fire-aware evaluator it hovered near 50% vs random.

## Candidate v1: Ceruledge Fire Efficient Attacker
- **Strategy**: Fast, consistent Stage 1 Fire attacker (Ceruledge #797).
- **Core Attack**: Infernal Slash (220 damage for 1 Fire Energy).
- **Key Supporters**:
  - Lillie's Determination (#1227): Draw 6 (or 8 if 6 prizes).
  - Judge (#1213): Hand disrupt + draw 4.
  - Cyrano (#1205): Search 3 Pokemon ex/basic.
  - Boss's Orders (#1182): Gust opponent's bench.
- **Key Items / Tools**:
  - Buddy-Buddy Poffin (#1086): Bench setup for Charcadet.
  - Maximum Belt (#1158): ACE SPEC (+50 damage vs ex).
- **Energy**: 31x Basic Fire Energy (#2).
