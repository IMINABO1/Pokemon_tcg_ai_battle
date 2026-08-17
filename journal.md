# Journal — Pokémon TCG AI Battle Challenge

Merged log of both workstreams (Belex: July groundwork + instrumentation;
Imi: August overhaul, ladder campaign, cluster handoff).

## References
- Simulation Category: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle (submissions closed 2026-08-16 23:59 UTC; games run until leaderboard converges ~Aug 31)
- Strategy Category (write-up): https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy (entry 2026-09-06, final 2026-09-13)
- Episode datasets index: https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index (~21.5 GB/day, 4.5–8k episodes/day)
- Beginner packaging guide: https://www.kaggle.com/code/ichigoe/beginner-guide-from-deck-list-to-first-valid-sub
- Docker image for cluster training: `docker.io/iminabo/field_zero:latest`
- Meta reference: `pokemon_tcg_meta_history.txt` (1999–2026 archetype history; its 2026 section matched real ladder decks card-for-card)

## Project in one line
A determinized-search (PIMC) agent over the competition's real C++ engine for the ladder, plus a mined-replay imitation-learning pipeline (FIELD-Zero) for the strategy write-up and a possible learned evaluator.

## Decisions (with rationale)

### 2026-07-13 — Heuristic + PIMC search first; RL later (PLAN v1)
PIMC is the proven approach for imperfect-information TCG bots on tight compute. Rejected: pure self-play RL first — documented risk of strategy cycling/divergence and multi-week infra before any ladder signal.

### 2026-07-13 — Sequential-only traversal within one search tree
Engine source shows the RNG lives on `Game` and all nodes of one `search_begin` tree share it (`Search.h:238`, `State.h:111`, `Game.h:61`). Any parallelism must be cross-process. Rejected: in-process threading of sibling branches (would corrupt the shared RNG stream).

### 2026-07-13 — Single-energy efficient-attacker deck (Ceruledge) over the sample deck — later reversed
Chosen because "attach, attack, repeat" seemed easiest for a shallow-search pilot. Reversed 2026-08-15 when Infernal Slash's hand-discard rider proved the deck whiff-prone (see problems log); the searcher turned out to be deck-agnostic because the engine prices attack riders inside rollouts.

### 2026-08-01 — Observability before fixes
Counters at every exception-swallow site (`stats.py`) before touching the search. They refuted the leading "search never runs" hypothesis in 30 minutes (see problems log) and redirected the effort to decision quality.

### 2026-08-15 — Rollouts run to a fixed *turn* horizon, not fixed decision depth
Fixed depth compared "attacked → opponent replied" leaves against "stalled → nobody replied" leaves at different game phases, so passive lines always won and the agent stopped attacking. Rejected: deeper fixed depth (same parity bug, slower); full expectimax (sequential-only engine, too slow).

### 2026-08-15 — Static develop-then-attack rollout policy instead of eval-greedy expansion
Greedy expansion evaluates every candidate at every rollout step (~10× cost). Static priority (EVOLVE > ATTACH > ABILITY > PLAY > ATTACK > … > RETREAT) is ~30× cheaper; attack ranks below development because attacking ends the turn.

### 2026-08-15 — Adaptive per-decision time budget from `remainingOverageTime`
The real ladder gives `actTimeout=0` plus a per-agent 600 s bank per game — DQ at 0 (answered PLAN v1's "biggest open unknown"). Budget = clamp(bank / expected-remaining-decisions, 0.3 s, 2.5 s), heuristic fallback under 40 s. Rejected: fixed 2.0 s/decision (would DQ ~300 decisions into a long game).

### 2026-08-15/16 — Deck evolution driven by replay evidence
v1 Ceruledge retired (whiff rider) → sample Kyogre/Mega Abomasnow shell (16/16 vs random and greedy locally) → v3 (+4 Kyogre vs anti-ex walls in the pool) → v4 (13 basics + benchless eval penalty, after ranked replays showed benchless insta-losses) → v5 = the exact ~1280-Elo Grimmsnarl/Munkidori list mined from top-team replays. Rejected: Buddy-Buddy Poffin engine for our shells (our basics all exceed its 70 HP limit).

### 2026-08-16 — Opponent determinization = mirror archetype prior minus visible cards, serial-deduped beliefs
Mirror-of-our-deck is the dominant-meta assumption and guarantees internally consistent worlds; visible-card subtraction plus per-name copy caps keeps samples legal. Belief dedup by card serial (a physical card is re-logged on every move — the un-deduped list grew without bound, plausibly Belex's observed late-game degradation).

### 2026-08-16 — Ship two final variants and let the ladder arbitrate (final rank = best submission)
v0.4a = v4 deck; v0.4b = mined Grimmsnarl list. Local suites saturate (both ≥15/16 vs all baselines), so the ladder was the only remaining discriminator. Verdict within hours: archetype, not compute, is the binding variable (identical search: ~510 vs ~760).

### 2026-08-16 — Training on Quest (Northwestern HPC) via Docker → Apptainer, not locally
No root on shared clusters → no dockerd; Apptainer consumes the same image unprivileged. Environment ships in the image (CUDA 12.4 *runtime* base — nothing compiles CUDA here, saves ~7 GB); code ships via git clone, bind-mounted over the image's baked copy so `git pull` beats rebuild+push.

### 2026-08-16 (~08:30 UTC) — Final submission slot HELD pending ladder evidence (owner decision)
Owner checked the leaderboard distribution first: 6,846 teams, bell curve, median 612, ceiling 1273 — the hoped-for 1700–2000 region does not exist (800 = top 14%, 900 = top 6%, 1000 = top 2%). v0.4b was climbing 604 → 725 at decision time. Decision: hold, watch v0.4b's plateau, decide near the deadline. A v6 candidate (mined Mega Lucario ex list — the deck that beat us twice) is benchmarked and ready: 14/16 vs Grimmsnarl-greedy, 16/16 mirror, 16/16 wall, worst game 84 s/600. RL/BC by tonight acknowledged as unrealistic (featurizer unwritten, data unmined).

### 2026-08-16 — Post-merge: ladder-validated code wins over instrumentation-era edits
Merging `origin/imi` grafted the 08-01 evaluator rebalance onto the overhauled agent (non-conflicting hunks apply even with `-X ours`). Restored the validated versions; kept `stats.py` and `test_evaluate.py`, with the hand-visibility invariant test xfail'd as a documented divergence (turn-horizon rollouts evaluate all sibling candidates at the same phase, so the visibility bias is shared).

## Open questions
- Spend the held final slot on the benchmarked v6 Lucario agent before 23:59 UTC, or hold it unused? Watch v0.4b's plateau (~750–790 as of ~12:00 UTC).
- Ratings for weighted imitation: replays carry no mu/sigma — join against Kaggle's episode-metadata tables, or accept `w_skill(None)=0.25`? Answer during bulk mining on Quest.
- Featurizer design for BC (the deliberate remaining seam in `bc_train.py`): one shared module for training and submission runtime.
- Whether the Strategy entry needs reproducibility artifacts attached (rules page is login-gated; owner to read the official text).
- Reconcile the hand-visibility evaluator invariant (xfail'd test) with the shipped visibility-dependent terms — revisit if leaf phases ever diverge.

## Progress log

### 2026-07-20 — Week-1 baseline on the ladder
Heuristic-only agent + placeholder mono-Water deck (sub 54863866) to start the ladder clock and measure timing. Settled at 344.4. Its episode logs later became the key evidence for the real timing model.

### 2026-07-30 — PIMC v1 and its bugs
First search agent (commit `e2e12ab`) fixed six bugs (COUNT returned the count value, not an index; YES/NO by position; etc.), but the recorded suite predated the fix and the fixed agent was never re-benchmarked.

### 2026-08-01 — Baseline established: pipeline works, agent does not (Belex)
Full architecture end-to-end; local suite 1/5 vs random. Code-read suggested the search never runs (swallowed exceptions → "always option 0").

### 2026-08-01 — Instrumentation refuted the "search never runs" hypothesis (Belex)
Counters at every swallow site: over 20 matches, 0 fallbacks, 0 `search_begin` failures, 17,532/17,532 rollouts scored. The search runs and discriminates (23.8% non-default picks; 23.7% degenerate ties). True signal: win/loss tracks game length — every win short (15–36 actions), every loss long (148–192). Redirected effort to evaluator quality, rollout depth, and belief drift — the same three defects independently confirmed and fixed on 2026-08-15.

### 2026-08-15 — Overhaul day (commit `a5a8b97`)
Re-benchmarked: 6/16 vs random, 11/16 vs greedy. Fixed in sequence: adaptive timing from the 600 s bank; rollout parity bug → turn-horizon rollouts (11/16 → 14/16 vs greedy); Infernal Slash whiff discovery → deck switch to sample shell (**16/16 vs random and greedy mirror**); determinization accounting (serial dedup, visible-card subtraction, working partial-legality gate). Gate suite 29/32 vs random, 32/32 vs greedy; worst game 11.7 s of the bank. Anti-ex wall hedge v3 (commit `ba6177a`).

### 2026-08-16 (early UTC) — Ship, crash, diagnose from the source
v0.2 ERRORED instantly: kaggle-environments `exec()`s `main.py` with no `__file__` and takes the last callable as the agent. Fixed (`a7dbd34`), verified by running the actual `cabt` env locally. v0.2.1 ranked 5-8 (~478): replays showed no crashes/timeouts but **benchless insta-losses** (one loss ended turn 3 with all prizes intact). Deck v4 (`3f8468f`) submitted as v0.3 → ~570 (top 58%). FIELD-Zero schema frozen from 5 real episodes (`8ce4e6a`): 806 decisions, 0 minCount violations, leakage audit exit 0. Ladder-walk scouting found the ~1280-Elo lists: an identical Grimmsnarl/Munkidori 60 across multiple top teams, and a 4× Crustle wall deck.

### 2026-08-16 (day) — Final submissions, ladder verdict, cluster handoff
- Search cranked (dets 16 → 32, budget cap 1.2 → 2.5 s), timed cabt self-play safe on both decks. Submitted **v0.4a** (v4 deck) and **v0.4b** (mined Grimmsnarl list).
- Verdict: v0.4a sank to ~510 while v0.4b climbed 604 → 785 peak, settling ~750–790 (top ~15% of 6,846) on identical search code. Both v0.4a losses analyzed: one more benchless death, one race lost to a Mega Lucario ex tank; Lucario list extracted as candidate v6 and benchmarked. **Final slot held** (see decision).
- Quest handoff: `iminabo/field_zero:latest` on Docker Hub; `field_zero/slurm/{mine_data,bc_train}.sbatch`; everything merged to `main` for the teammate.
- Repo reconciliation: `origin/imi` merged and pushed (`2e4fbf3`) after restoring ladder-validated agent code; README added; worklogs merged into this file (tracked going forward, superseding the earlier gitignore-them decision since the remote already tracked them).

### 2026-08-16 (evening UTC) — Elo-band study (owner's idea: study the bands above us before spending the last slot)
- Collected 117 public replays via ladder-walk: 50 from the 800 band, 20 from 900, 47 from 1000.
- **Each band has a different king**: 800 = Grimmsnarl (59% share, most wins); 900 = Mega Lucario (48%, 12/21 winner decks); 1000 = **Dragapult ex (64% share, dominant winners)** — matching the meta-history doc's top archetype. Extracted each band's winningest 60-card list; the 1000-band Dragapult list saved as candidate v7.
- Playstyle gap vs us (even in our wins): their avg bench 3.1–3.4 and benchless-turn rate ~2%, ours 1.9 and 10%; their first attack turn ~3.6–4.6, ours ~5.1. Benching speed remains our biggest behavioral gap — write-up material and future evaluator target.
- Gauntlet (v5 Grimmsnarl / v6 Lucario / v7 Dragapult × three band-winner decks) invalidated once by rig oversubscription (see problems log), then rerun clean: **v5 24/24, v7 22/24, v6 21/24** (v6's losses concentrated vs the 1000-band Dragapult deck).
- Extended census to 600/650/700 bands (150 more replays) for the owner's "is there a low-band Dragapult trap?" question. Answer: no — sub-800 is 33-68% Grimmsnarl (largely public-notebook forks; a 984-vote sample-agent family feeds the ladder); Dragapult is rare below 1000 (6-15%) with ~50% winrates and near-reference play metrics, i.e. in transit, not stuck. Band ceilings visible in the data: Grimmsnarl ~850, Lucario ~950, Dragapult = the only deck winning at 1000+ (64% share, 70% of wins). Decks are frozen per submission, so the deck choice IS the ceiling choice.
- **Post-deadline twist (owner spotted it):** Kaggle opened an official Late Submission window at the deadline (quota jumped to 100/day; organizers emailed the team). v0.6 = the 900-band Mega Lucario list submitted 00:21 UTC Aug 17, validated COMPLETE at 600 with a private score assigned — completing the three-ceiling portfolio (Grimmsnarl ~850 / Lucario ~950 / Dragapult summit) after all. Whether late entries count for final rank awaits the email's exact wording; worst case v0.6 is sanctioned live benchmarking for the write-up.
- **Final slot: v0.5 submitted with owner approval at ~23:25 UTC** — v7 Dragapult (the 1000-band's winningest list: Dreepy/Drakloak ×4, Dragapult ex ×3) on the unchanged validated searcher. Pre-submit scare: an over-strict assert (demanded 4× Dragapult ex; the real list runs 3) fired while PowerShell's `;` chaining submitted anyway — post-hoc byte-compare confirmed staging = candidate v7 exactly, submission correct. Games continue until ~Aug 31; final rank = best of {v0.4b ~758, v0.5, older subs}.
