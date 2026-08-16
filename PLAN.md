# Pokémon TCG AI Battle — PLAN v2 (Fix, Strengthen, Win)

Supersedes PLAN v1 (2026-07-13, in git history). v1 planned *building* a heuristic+PIMC agent on a
mistaken 4-week runway; that architecture is now **built** and the real runway is ~6 weeks. What
changed: (1) the true deadlines are a month later than v1 assumed, (2) the built agent currently
**loses 4/5 to a random agent** and we know exactly why (Section 4), (3) the Strategy Category
writeup — a separately judged deliverable — does not exist yet. This plan is organized around those
three facts.

## 1. Mission & scoring reality

Win **both** coupled categories (Strategy entry requires Simulation entry):

- **Simulation Category**: ladder win rate via continuous matchmaking + Gaussian skill rating.
- **Strategy Category** (judged): Model Score **70%** (clarity of approach + rationale, originality
  and technical soundness, consistency under repeated matches, robustness across matchups/initial
  states, track performance), Deck Score **20%** (concept clarity, card choices supporting the game
  plan), Report Score **10%** (structure, effective figures/tables). Writeup ≤ 2000 words + optional
  media gallery.

Implication: evidence-driven development — hypothesis → experiment → numbers → decision — *is* the
writeup content, not overhead. `journal.md` / `problems_encountered.MD` (worklog discipline) are the
collection mechanism; the writeup is assembled from them, not written cold in September.

## 2. Timeline (absolute)

- Today: **2026-08-01**
- Entry / team-merger deadline: **2026-09-06** (both categories)
- Final submission: **2026-09-13**
- Judging: Sep 14 – Oct 11

~6 weeks. Team: 2–3 people, ~20–40 combined hours/week. Goal is to win, so the extra month v1
didn't know about goes to the RL track (Section 7) — the bet most teams won't successfully make —
not to slack.

## 3. Key engine facts (verified by direct source reading; carried from v1)

- `search_begin`/`search_step`/`search_end`/`search_release` (`cg/api.py`) are backed by a real
  clone-and-step C++ tree (`Search.h`), not a mock.
- **RNG is shared across every node of one search tree.** `Search::alloc()` does `*state = src`
  (`Search.h:238`), a shallow copy; `State::game` is a raw `Game*` pointer (`State.h:111`); the RNG
  (`std::mt19937 rng`, `Game.h:61`) lives on `Game`. All nodes descending from one `search_begin`
  share one RNG stream → **sequential-only traversal within one tree; never thread sibling branches
  of the same search**. Parallelism across the 2 vCPUs must be cross-process.
- `search_step` **auto-advances through no-choice states** (`Search.h:187-190`) — one call can
  silently fast-forward through several sub-decisions, and the next prompt may belong to the
  opponent. Never assume the returned prompt is the one you expected.
- `SelectContext` is explicitly unreliable per engine comments — branch only on
  `SelectType`/`Option.type`; treat `SelectContext` as a debug hint.
- `search_begin` requires a **complete** guess (rejects an unresolved face-down opponent active,
  error 98, checked in `Search.h:96-98` and pre-checked in `api.py`).
- Mid-search reshuffle (`Search::shuffle`, `Search.h:196-211`) is **not exported** to Python; live
  without it (a fresh `search_begin` per decision re-samples the determinization anyway).
- `cg/game.py`'s `Battle` holds `battle_ptr` as a **class-level singleton** — one local battle per
  Python process; the harness must use multiprocessing, never threads.
- Deck legality (`Api.h`, `Core.h`): exactly 60 cards; max 4 copies per name **except** Basic Energy
  (unlimited); max 1 ACE SPEC; ≥1 Basic Pokémon. Specific error codes per violation.
- 3000-action hard cap per game (`BattleData.h`); **no documented per-move timeout** — still the
  biggest open unknown; probe empirically via real submissions (Section 6, Track A).
- Canonical runtime card data: `cg.api.all_card_data()`/`all_attack()` — not `EN_Card_Data.csv`
  (offline research artifact only).

## 4. Current state (honest, as of 2026-08-01)

**Built and working:** ~1,060 LOC agent (`agent/src/ptcg_agent/`: policy, search, evaluate,
determinize, carddb, legality, budget, config, logging_utils); multiprocess local harness
(`agent/harness/`); passing unit tests; legal 60-card Ceruledge deck (`agent/decks/candidate_v1.csv`);
packaging + validation scripts; built `submission.tar.gz`; one real Kaggle episode (87163505).

**Broken:** the agent wins only **~25% vs Random** (5/20 measured, `agent/harness/results/suite_results.csv`).

> **Update 2026-08-01 (instrumented):** findings 1 and 2 below were a code-read *hypothesis* and are
> **refuted by measurement**. With counters on every swallow site: 0 whole-decision fallbacks, 0
> `search_begin` failures, 17,532/17,532 candidate rollouts scored, 0 rollout exceptions; the search
> discriminates (23.8% of decisions pick a non-default action, 23.7% are degenerate ties). The search
> runs fully — the agent loses on **decision quality**, and **win/loss tracks game length** (wins
> 15–36 actions, losses 148–192). Real culprits are findings 3, 4, 8, 6, in that priority. Findings
> 1–2 are kept below as a record of what was ruled out. See [[problems_encountered]].

Original code-read diagnosis (2026-08-01), ranked:

1. **[RULED OUT] Search never actually runs.** Matches of 42–125 actions finished in 0.63–2.06 s *total*, vs
   `PER_DECISION_BUDGET_SECONDS = 2.0` × `NUM_DETERMINIZATIONS = 6` (`config.py:3-5`). Every failure
   path is silently swallowed with no counter: per-candidate `except: pass` (`search.py:164-165`),
   per-determinization `except: continue` (`search.py:144-147`), whole-decision fallback
   (`policy.py:130-136`). When all rollouts fail, every average stays −1e9 and the code returns
   `candidates[0]` (`search.py:175,181`) — a deterministic "always option 0" policy, strictly worse
   than random, and *indistinguishable from a working search from the outside*.
2. **[RULED OUT] Likely root exception:** `_rollout` re-steps from the same `root_state.searchId` for every
   candidate (`search.py:77`); if the engine consumes the parent state on step, candidates 1..n die
   ("Released item"). `search_release` is imported but never called (`search.py:9`).
3. **Eval is information-dependent, punishing turn-ending.** Leaves are evaluated regardless of
   whose prompt is showing (`search.py:97-99`); after attack/END the observation is the opponent's,
   `me.hand is None`, so `_score_hand` (`evaluate.py:140-149`) drops the +8 supporter term — keeping
   the turn always outscores attacking. (Orientation and terminal signs verified correct:
   `evaluate.py:174-184`.)
4. **Eval weights reward hoarding.** `supporter_in_hand = +8`, `hand_size = +2` (`config.py:25-26`):
   playing a supporter costs −10 with no offsetting term; `energy_attached = 10` vs `hp_swing = 0.1`
   values one energy attachment equal to 100 damage dealt.
5. **COUNT fast-path returns a count as an index** (`policy.py:100-102` returns `[select.maxCount]`)
   → engine error 5 → instant forfeit in the match loop (`local_match.py:88-93`). `YES_NO → [0]` and
   `SPECIAL_CONDITION → [0]` unconditionally (`policy.py:95-97,105-106`).
6. **Degenerate determinization.** Hardcoded 44-card filler + pad-with-Basic-Fire-id-2 can produce
   illegal decks → `search_begin` raises → feeds finding 1 (`determinize.py:92-94,145-153,176-180`).
   `update_from_logs` re-scans all logs every decision and appends without dedupe → opponent belief
   becomes nonsense mid-game (`determinize.py:105-116`). `legality.py` exists but is unused here.
7. **Candidate enumeration truncation.** Only the first 8 options are kept (`search.py:43-46`) —
   ATTACK/END can be truncated off a busy MAIN prompt; multi-selects get the single candidate
   `list(range(max_c))` (`search.py:56-60`); decline-`[]` is always tried first (`search.py:49-53`).
8. **Rollout quality.** Depth 3 prompts is less than one PTCG turn (`config.py:5`); continuation is
   `candidates[0]` for *both* players (`search.py:90`); budget expiry starves later candidates
   (`search.py:151-152`).

## 5. Phase 0 — Make the search *strong* (Aug 1 → ~Aug 10) · critical path · strongest coder

**Step 0 is done** (2026-08-01): observability counters (`stats.py`) surfaced into the harness CSV.
They proved the search runs fully and reordered everything below — the plumbing fixes (old items 1)
are deprioritized because the failures they target do not occur. Order now runs by measured impact,
each fix verified against the counters (degenerate-tie rate, fallback rate) and win rate vs Random:

1. **Evaluator (highest impact).** Rebalance `EVAL_WEIGHTS`: remove the hoarding incentive
   (`supporter_in_hand +8`, `hand_size +2` make playing a supporter cost −10), raise `hp_swing`
   relative to `energy_attached` (currently 0.1 vs 10 → one energy = 100 damage), fix the own-prize
   double-count. Make the leaf term observability-invariant (opponent-perspective leaves have
   `me.hand is None`, silently dropping the +8 supporter term and biasing against ending the turn) —
   evaluate only at own-perspective prompts or guard the hand term. Target: degenerate-tie rate
   (baseline 23.7%) drops materially.
2. **Rollout quality.** Depth 3 prompts is < one PTCG turn; deepen to end-of-own-turn. Opponent
   continuation should be greedy-by-eval from the opponent's perspective, not `candidates[0]`.
3. **Opponent-belief drift (long-game collapse).** `update_from_logs` re-scans and re-appends every
   decision without dedupe/reset (`determinize.py:105-116`) → belief degrades as games lengthen,
   matching the observed "loses long games" pattern. Dedupe/reset; legality-check sampled worlds via
   the unused `legality.py`.
4. **Fast-paths & enumeration (correctness backstops).** Fix COUNT (returns a count where an index
   is expected — `policy.py:100-102`), YES_NO, SPECIAL_CONDITION via 1-ply evaluate; stop truncating
   ATTACK/END off busy MAIN prompts and privileging decline-`[]`. (Not currently forfeiting in the
   Ceruledge-vs-sample suite, but latent.)
5. **[deprioritized] Rollout state lifecycle** (old item 1): call `search_release` for hygiene, but
   the "candidates 1..n error out" failure was measured at zero — not a strength lever.

**Gate G0:** ≥90% vs Random over ≥100 multiprocess matches, AND the search agent beats the
heuristic-only (no-search) agent head-to-head. Numbers logged to `journal.md`.

## 6. Phase 1 — Strength & ladder (Aug 10 → Aug 29) · parallel tracks

- **Track A — agent strength.** Probe the undocumented per-move timeout with real submissions early;
  keep a legal submission on the ladder **at all times** (the rating clock only runs while
  submitted). Cross-process PIMC parallelism (one determinization per OS process, 2 vCPUs).
  Opponent-response modeling in rollouts. Eval-weight autotuning on the cluster's CPUs: perturb →
  fixed match budget vs a fixed opponent pool → keep/discard (the v1 §7 hook; `EVAL_WEIGHTS` is
  already a flat dict).
- **Track B — deck.** 3–5 candidate decks vs `candidate_v1` (Ceruledge); round-robin on the
  now-trustworthy harness; ≥200 matches per pairing before believing a difference. Deck rationale
  recorded as it happens → this *is* the Deck Score (20%) content.
- **Track C — Strategy writeup (continuous, starts now).** Maintain `journal.md` /
  `problems_encountered.MD` per the worklog skill. Build the Kaggle notebook rendering harness CSVs
  (win-rate curves, matchup tables, fallback-rate-over-time — the "broken search was invisible"
  story is itself strong writeup material). Draft the ≤2000-word writeup by **Aug 29** so September
  is results-refresh only. Enter both categories on Kaggle well before Sep 6.

## 7. Phase 2 — RL value net (Aug 15 → Sep 5, overlaps Phase 1) · 2nd/3rd teammate + GPU cluster

The "win" bet: a learned evaluator inside real-engine PIMC search — most teams will ship either pure
heuristics or pure RL; the combination is the edge.

- Self-play data generation reuses the harness match driver; per-decision logging
  (`logging_utils.py`) already emits (state, action, outcome) JSONL.
- The net swaps into the `evaluate_state` seam — `search.py` accepts `evaluator=` (v1 §7 hook,
  already built); search/policy control flow untouched.
- Anti-cycling from day one: frozen-snapshot opponent pool (Lux AI S1 / NFSP precedent); never naive
  vanilla self-play.
- **Gate G2 (Sep 3):** the RL evaluator must beat the tuned-heuristic agent ≥55% over ≥300 local
  matches to ship. Otherwise the heuristic ships — the gate keeps the win-bet from becoming a
  lose-bet.

## 8. Freeze & submit

- **Sep 5:** lock deck + agent candidate.
- **Sep 6:** entry/merger deadline (already entered; hard backstop).
- **Sep 6–12:** validation only — `validate_submission.py` against the packaged artifact, packaged
  matches, edge-case sweep (minCount 0, forced discards, COUNT selects, `select is None` deck
  request). Finalize writeup + media gallery. No risky changes.
- **Sep 13:** final submissions, both categories.

## 9. Risks

Carried from v1 (still live): undocumented per-move timeout (probe early, conservative budget);
shared RNG per tree (sequential traversal only); `SelectContext` unreliable (key off `SelectType`);
`Battle` singleton (multiprocess harness); own-state bookkeeping corrupting `search_begin` (unit
tests).

New in v2:
- **Silent-fallback regression** — the Phase 0 counters stay in as permanent regression checks;
  fallback rate is reported in every harness run and watched in every A/B.
- **RL non-convergence or cycling** — gate G2 + frozen-snapshot pool; heuristic agent is never at
  risk as the shipped fallback.
- **Autotune overfitting** to Random/self — tune against a fixed, diverse opponent pool (random,
  heuristic-only, prior snapshots, alternate decks).
- **Single-deck metagame blindness** — Track B breadth + watching ladder replays for common
  archetypes.

## 10. Team & cadence

- **P1 (strongest coder):** Phase 0 → Track A.
- **P2:** Track B → Phase 2 RL ownership.
- **P3 (or rotating):** Track C + harness/results ops.
- `journal.md` is the sync point (decisions with rationale + numbers); `problems_encountered.MD` for
  substantive problems only, per the worklog skill. Weekly checkpoint against gates G0/G2; if G0
  slips past Aug 12, Phase 2 start slips with it — the RL track depends on a trustworthy harness
  signal, not calendar dates.
