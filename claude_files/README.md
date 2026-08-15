# FIELD-Zero — Stage 0: Episode Mining + Imitation Foundation

Field-Imitation and Explicit-Latent Decision Search. This package is the
first-class foundation everything else (BC → league PPO → adversarial
mining → search+PRISM) depends on.

## Pipeline order

```
1. python mining/fetch_episodes.py          # pull daily episode datasets via Kaggle API
2. python mining/inspect_replay.py <file>   # FREEZE THE SCHEMA FROM REALITY FIRST
3. python mining/parse_replay.py            # replay JSON -> per-player decision records
4. python mining/build_dataset.py           # -> games.parquet + decisions.parquet (+ weights)
5. python mining/audit_leakage.py           # hard-fail if any forbidden field is a model input
6. python model/bc_train.py                 # behavior-clone the pointer policy from the field
```

## Non-negotiable rules (enforced in code, not vibes)

- **Leakage rule**: a feature is a legal model input iff our own agent could
  compute it at that exact decision point from its own observation stream.
  `audit_leakage.py` hard-fails the build otherwise. Opponent's revealed-at-end
  deck/archetype are AUXILIARY TARGETS only, never inputs.
- **Schema from reality**: `parse_replay.py` has explicit `# ADAPT:` markers.
  Run `inspect_replay.py` on one real replay and fix those markers BEFORE
  bulk-parsing. Do not trust assumed JSON shapes.
- **Weighted imitation**: skill x confidence x recency x diversity weights are
  computed in `build_dataset.py` and stored per-row. BC must consume them.
- **Pointer policy**: the network scores each legal option; no fixed action
  vocabulary. New deck != new output layer.

## Requirements

Dev machine only (never the submission tarball): `kaggle`, `pandas`,
`pyarrow`, `numpy`, `torch`. Kaggle API token at `~/.kaggle/kaggle.json`.

## Where things land

```
data_lake/raw/<date>/        raw episode JSONs per daily dataset
data_lake/parsed/            one JSONL of decision records per game
data_lake/games.parquet      one row per game
data_lake/decisions.parquet  one row per (player, decision)
```
