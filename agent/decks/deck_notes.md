# Deck Candidates & Performance Notes

## Candidate v4 (LIVE): +Basics package (Suicune/Glastrier)
- **Change from v3**: -4 Water Energy (35->31), -1 Mega Signal (2 left),
  +3 Suicune #803, +2 Glastrier #867. Basics: 8 -> 13.
- **Why**: Ranked replays of v0.2.1 (submission 55538879, ~478 Elo, 5-8)
  showed repeated benchless insta-losses: with 8 basics the agent often
  never benches, so one KO ends the game. Search scored these positions
  correctly (-1e6 in half of sampled worlds) - the deck was creating dead
  positions. Buddy-Buddy Poffin can't help (our basics all exceed 70 HP),
  so the fix is more self-sufficient basics: Suicune (30+90 with 4 {W} in
  play), Glastrier (1-energy 20+20 snipe, 130 for {W}{W}{W}).
- **Paired change**: EVAL benchless_penalty -40 + bench_count 8 so the
  evaluator prices bench insurance beyond the rollout horizon.
- **Results**: 16/16 vs greedy piloting v3 (meta proxy), 16/16 vs wall,
  27/32 vs random; max in-process decision 1.2s.

## Candidate v3 (LIVE): Sample Water + 4 Kyogre
- **Change from v2**: +2 Kyogre (2→4), -1 Cyrano (2→1), -1 Mega Signal (4→3).
- **Why**: The card pool contains anti-ex walls (Sylveon #330 Safeguard,
  Crustle #345 Mysterious Rock Inn, Neutralization Zone #1247) that fully
  blank Mega Abomasnow ex. Kyogre is the non-ex answer (Swirling Waves 130
  OHKOs Sylveon), dodges opposing Maximum Belt (+50 is vs ex only), and
  concedes 1 prize instead of 3 when KO'd. Hammer-lanche's self-mill also
  feeds Riptide's water-in-discard scaling.
- **Results**: 16/16 vs random, 16/16 vs sample-greedy mirror, 16/16 vs a
  Sylveon/Crustle wall deck (test_wall.csv) piloted by greedy.

## Candidate v2: Sample Water Deck (Kyogre / Mega Abomasnow ex)
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
