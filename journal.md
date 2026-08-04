# Journal

## References
- Simulation Category: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Strategy Category: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy
- Beginner packaging guide (submission.tar.gz format): https://www.kaggle.com/code/ichigoe/beginner-guide-from-deck-list-to-first-valid-sub

## Project in one line
PIMC (guess opponent's hidden cards, search each guess in the real C++ engine, average) + heuristic
evaluator agent for the Pokémon TCG AI Battle Challenge, targeting wins in both the Simulation and
Strategy categories.

## Decisions (with rationale)

### 2026-07-13 — Heuristic + PIMC search first; RL later (from PLAN v1)
Chose a hand-built evaluator inside PIMC search over starting with self-play RL. Why: PIMC is the
proven approach for imperfect-information TCG bots on tight compute, and the plan was drafted
believing only ~4 weeks remained. Rejected: pure self-play RL first — documented risk of strategy
cycling/divergence (Lux AI S1, NFSP precedent) and multi-week infra before any ladder signal.

### 2026-07-13 — Single-energy efficient-attacker deck (Ceruledge) over the sample mill deck
"Attach, attack for near-lethal, repeat" is far easier for a heuristic+shallow-search pilot to play
correctly than combo/mill sequencing; early on, play quality is the bottleneck, not deck power.
Rejected: adapting the sample mono-Water mill deck (requires precise sequencing the agent can't
deliver yet).

### 2026-07-13 — Sequential-only traversal within one search tree
Engine source shows the RNG lives on `Game` and all nodes of one `search_begin` tree share it
(`Search.h:238`, `State.h:111`, `Game.h:61`). Any parallelism must be cross-process. Rejected:
in-process threading of sibling branches (would corrupt the shared RNG stream).

### 2026-08-01 — Rewrite PLAN.md as v2 rather than patch it
Real deadlines are entry 2026-09-06 / final 2026-09-13 — a month later than v1 assumed — and v1
describes building an architecture that is already built (~1,060 LOC, working harness, passing
tests, packaged submission, one live Kaggle episode). Rejected: in-place date edits (most of the
document planned work that's done) and keeping a parallel PLAN_V2.md (two plans invite drift; git
history preserves v1).

### 2026-08-01 — RL value-net track is ON, gated
School GPU cluster is available and a teammate can own it. The learned-evaluator-inside-real-engine-
PIMC combination is the "win" bet most teams won't make. Gated (G2, Sep 3): must beat the tuned
heuristic ≥55% over ≥300 local matches or the heuristic ships. Rejected: search-only for safety
(goal is to win, and the recovered month funds the attempt) and ungated RL (shipping on faith is the
documented failure mode).

## Open questions
- **Per-move timeout on Kaggle** — undocumented anywhere in the provided source; the single biggest
  unknown. Answer by: early real submissions with per-decision wall-clock logging (Track A).
- **Why exactly does every rollout fail?** Prime suspect is re-stepping from a consumed
  `root_state.searchId` (`search.py:77`). Answer by: Phase 0 step 0 counters, then step 1 fix.
- **Is Ceruledge actually the right deck?** Zero empirical evidence yet. Answer by: Track B
  round-robin, ≥200 matches per pairing, once the pilot is trustworthy (post-G0).

## Progress log

### 2026-08-01 — Baseline established: pipeline works, agent does not
- Full architecture is implemented and end-to-end: agent modules, multiprocess harness, unit tests
  passing, legal Ceruledge deck, packaged `submission.tar.gz`, one real Kaggle episode (87163505,
  self-mirror validation match).
- Local suite result: **1 win / 5 matches vs a random agent**
  (`agent/harness/results/suite_results.csv`).
- Code-read diagnosis: matches of 42–125 actions completed in 0.63–2.06 s *total* against a
  configured 2.0 s/decision × 6 determinizations — the search never actually runs. All exception
  paths are silently swallowed and the code degrades to "always pick option 0", which is worse than
  random. Seven further defects ranked in PLAN.md §4 (eval weights that pay −10 to play a supporter,
  a COUNT handler that returns a count where an index is expected → instant forfeit, degenerate
  opponent-belief sampling, ATTACK/END truncated from candidate lists).
- Interpretation: this is a systematic-defect problem, not a strength-tuning problem. Phase 0 of
  PLAN v2 is the ordered fix list; gate G0 = ≥90% vs Random over ≥100 matches.

### 2026-08-01 — Instrumentation refuted the "search never runs" hypothesis
Built Phase 0 observability first (`agent/src/ptcg_agent/stats.py`: counters at every swallowed
exception site, surfaced into `agent/harness/results/suite_results.csv`). Ran a 20-match baseline
vs Random:

| Metric | Value |
|---|---|
| Win rate vs Random | **5/20 = 25%** |
| Whole-decision fallbacks | 0 |
| `search_begin` fail | 0 / 3774 |
| Candidate rollouts scored | 17532 / 17532 (100%) |
| Rollout exceptions | 0 |
| Degenerate ties (all cands equal) | 149 / 629 search decisions (23.7%) |
| Picked non-default action | 150 / 629 (23.8%) |

- **The search runs fully and discriminates** — the earlier "silently degrades to option 0"
  root-cause guess is **wrong** (see [[problems_encountered]] entry). The old ~0.6–2 s timing that
  looked "too fast to be searching" was just a shallow, cheap rollout; it *is* searching.
- Strong signal instead: **win/loss tracks game length** — PIMC wins were 15–36 actions, losses
  148–192. Play degrades over long games.
- **Direction change for Phase 0:** deprioritize the rollout state-lifecycle rewrite (it fixes a
  failure that isn't occurring); prioritize (1) evaluator weight rebalance + tie-breaking, (2)
  rollout depth / opponent-continuation quality, (3) opponent-belief drift over long games. The
  degenerate-tie rate and fallback rate are now the regression signals to watch per fix.
- All 23 unit tests still pass; instrumentation is behavior-neutral. (Installed `pytest` into the
  fresh venv to run them.)
