# Pokémon TCG AI Battle Challenge

Agent for The Pokémon Company's [PTCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
on Kaggle (Simulation category, closed 2026-08-16), plus a replay-mining and
imitation-learning pipeline feeding the
[Strategy category](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)
write-up (due 2026-09-13).

## What the agent is

A determinized-search (PIMC) agent over the competition's real C++ engine — no
trained model at decision time:

1. **Determinize**: sample plausible completions of hidden information
   (opponent hand/deck/prizes) from a mirror-archetype prior, minus every card
   already seen, with per-name copy caps. Own hidden zones are exact
   set-subtraction against our decklist.
2. **Search**: for each candidate action, `search_begin`/`search_step` the real
   engine forward — so attack riders, coin flips, evolutions, and abilities are
   priced by the actual rules, not approximations.
3. **Rollout to a turn horizon**: play both sides with a static
   develop-then-attack policy until the start of our next turn, so every
   candidate line is evaluated at the same game phase (fixed-depth rollouts
   made passive lines look better than attacking).
4. **Evaluate & average**: score leaves with a hand-tuned evaluator
   (prizes >> lethal threats >> board/tempo terms), average across
   determinizations, play the argmax.

Timing is governed by the ladder's real constraint — a per-agent 600 s
thinking bank per game (`remainingOverageTime`) — via an adaptive per-decision
budget with a heuristic fallback when the bank runs low.

## Results (ladder, 6,846 teams)

| Version | Deck | Public score |
|---|---|---|
| v0.1 heuristic | mono-Water placeholder | 344 |
| v0.3 | Kyogre/Mega Abomasnow + Suicune basics | ~570 (top 58%) |
| **v0.4b** | mined top-meta Grimmsnarl/Munkidori list | **~760 (top ~15%)** |

The decisive variable was deck archetype, not search strength: identical code
scored ~510 with the Suicune shell and ~760 with the meta Poffin/evolution-ex
shell mined from top-team replays.

## Repository layout

```
agent/
  src/ptcg_agent/     runtime package: policy, search, determinize, evaluate,
                      carddb, legality, config, deck.csv
  harness/            local match runner + benchmark suites (random/greedy
                      baselines, seat alternation, 600s-bank simulation)
  scripts/            build_submission.py, validate_submission.py
  decks/              candidate decklists + notes (deck_notes.md)
  tests/              pytest suite
field_zero/           FIELD-Zero: replay mining -> dataset -> BC/PPO training
  mining/             fetch_episodes, parse_replay, build_dataset, audit_leakage
  model/              pointer_policy, bc_train, ppo_league
  slurm/              Quest (HPC) job scripts; Dockerfile for the training image
sample_submission/    competition-provided engine bindings (cg/) + reference agent
ptcg_engine/          competition-provided C++ engine headers (reference)
EN_Card_Data.csv      card database (offline reference; runtime truth is cg.api)
```

## Quickstart (local)

```bash
python -m pytest agent/tests                 # unit tests
python agent/harness/smoke_match.py          # one PIMC-vs-greedy game
python agent/harness/run_matches.py --n 16 --opponent greedy   # benchmark suite
python agent/scripts/build_submission.py     # -> agent/dist/submission.tar.gz
python agent/scripts/validate_submission.py  # structure + packaged-agent match
```

Env knobs: `PTCG_DETS` (determinizations), `PTCG_BUDGET` (per-decision
seconds), `PTCG_DECK` / `PTCG_OPP_DECK` (deck CSV overrides for the harness).

Before submitting, also run a real `kaggle-environments` episode — the runner
`exec()`s `main.py` with no `__file__` and takes the last callable as the
agent, which local imports won't catch:

```bash
python -c "from kaggle_environments import make; e = make('cabt'); \
  e.run(['agent/dist/staging/main.py']*2); print([s.status for s in e.state])"
```

## Training on the cluster (FIELD-Zero)

Environment ships as a Docker image (`iminabo/field_zero:latest`); code ships
via this repo. On Quest (or any Apptainer HPC):

```bash
git clone https://github.com/IMINABO1/Pokemon_tcg_ai_battle
cd Pokemon_tcg_ai_battle/field_zero
module load apptainer
apptainer pull field-zero.sif docker://iminabo/field_zero:latest
# fill in --account/--partition in slurm/*.sbatch, put your kaggle.json at ~/.kaggle/
sbatch slurm/mine_data.sbatch    # episodes -> decisions.parquet (CPU)
sbatch slurm/bc_train.sbatch     # behavior cloning (GPU)
```

`bc_train.py` intentionally refuses to run until the shared featurizer
(observation -> tensors, used identically at train and inference time) is
written — that seam is the next piece of work.
