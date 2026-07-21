# Pokémon TCG AI Battle — Phase 1 Implementation Plan (Heuristic + PIMC Search)

## Context

Competing in "The Pokémon Company - PTCG AI Battle Challenge Simulation" (Kaggle), a two-player
imperfect-information Pokémon TCG ladder judged by continuous matchmaking + Gaussian skill rating.
Goal: **top 10 of ~5,000 teams** (99.8th percentile) — not just "ship something that works."
Timeline: today 2026-07-13; entry/merger deadline 2026-08-09; final submission 2026-08-16; ladder
converges ~2026-08-31.

Deep research this session (engine source analysis + a verified web research pass) established:
naive single-determinization search is theoretically exploitable, but **PIMC/ISMCTS-style search
(sample plausible opponent hands, search each, average)** is the practical, provably-used approach
in real TCG bots (e.g. Hearthstone AI Competition). Pure self-play RL is powerful but risks
strategy-cycling/divergence (confirmed via Lux AI Season 1 and NFSP precedent) and is a multi-week
infra project — too risky to attempt cold with a 4-week runway. The user explicitly chose to ship a
**heuristic + PIMC search agent first**, fast, and only later (if time remains) swap the hand-built
evaluator for a learned value net trained via self-play on the school GPU cluster, reusing the same
search scaffold. This plan covers that first phase end-to-end, with explicit hooks for the later swap.

The competition provides a genuine forward-model API (`cg.api.search_begin`/`search_step`) — hand
the real C++ engine a hypothesized full game state (including a guessed determinization of the
opponent's hidden cards) and step the actual rules engine forward. This is the centerpiece the whole
architecture is built around.

## Key engine facts (confirmed by direct source reading + independently re-verified this session)

- `search_begin`/`search_step`/`search_end`/`search_release` (`cg/api.py`) are backed by a real
  clone-and-step C++ tree (`Search.h`), not a mock.
- **RNG is shared across every node of one search tree.** `Search::alloc()` does `*state = src`
  (`Search.h:238`), a shallow copy; `State::game` is a raw `Game*` pointer (`State.h:111`); the RNG
  (`std::mt19937 rng`, `Game.h:61`) lives on `Game`. So all nodes descending from one `search_begin`
  call share one RNG stream → **sequential-only traversal within one tree; never thread sibling
  branches of the same search**. Parallelism across the 2 vCPUs must be cross-process (separate
  `agent_ptr`/`Game` per OS process).
- `search_step` **auto-advances through no-choice states** (`Search.h:187-190` loop calling
  `state.step()` while `selectMax == 0`) — a single call can silently fast-forward through several
  sub-decisions. Never assume the next returned prompt is the one you expected.
- `SelectContext` is explicitly commented in the engine source as unreliable — **branch only on
  `SelectType`/`Option.type`**, treat `SelectContext` as a log/debug hint only.
- `search_begin` requires a **complete** guess (rejects an unresolved face-down opponent active,
  error code 98, checked both in C++ `Search.h:96-98` and pre-checked in `api.py`).
- The C++ engine supports mid-search deck reshuffling (`Search::shuffle`, `Search.h:196-211`) but
  this is **not exported** to Python (missing from `Export.cpp` and `sim.py`) — a real capability
  gap; live without it (fresh `search_begin` per decision already re-samples the determinization).
- `cg/game.py`'s `Battle` class holds `battle_ptr` as a **class-level singleton** — only one local
  battle can run per Python process; the local test harness must use multiprocessing, not threads.
- Deck legality (`Api.h` `ApiBattleStart`, `Core.h`): exactly 60 cards; max 4 copies per named card
  **except** Basic Energy (unlimited); max 1 ACE SPEC; at least 1 Basic Pokémon required. Specific
  error codes exist for each violation.
- 3000-action hard cap per game (`BattleData.h`); no per-move "thinking time" timeout is documented
  anywhere in the provided source — **this is the single biggest open unknown** and must be probed
  empirically as early as possible via a real Kaggle submission.
- Canonical runtime card data source: `cg.api.all_card_data()`/`all_attack()` (from the same
  compiled binary the match runs on), not `EN_Card_Data.csv` (a secondary/derived research artifact).

## 1. Repo scaffolding

Create a new `agent/` directory as a sibling to the existing read-only competition folders
(`ptcg_engine/`, `sample_submission/`, CSVs/PDFs) — a git-tracked dev workspace, kept separate from
provided files.

```
pokemon-tcg-ai-battle/
├── ptcg_engine/, sample_submission/, *.csv, *.pdf   # read-only, competition-provided
└── agent/                                            # NEW dev workspace
    ├── pyproject.toml                                # zero/stdlib runtime deps; pytest/pandas under [dev] only
    ├── cg/ -> symlink to sample_submission/sample_submission/cg   # dev convenience only
    ├── src/ptcg_agent/
    │   ├── main.py            # thin Kaggle entrypoint: read_deck_csv() / delegate to policy.agent_decide
    │   ├── deck.csv            # current candidate deck, source of truth
    │   ├── carddb.py           # card knowledge base (Section 2)
    │   ├── determinize.py      # own-state tracker + opponent belief + sampling (Section 4.2)
    │   ├── search.py           # PIMC rollout + root-averaging (Section 4.3)
    │   ├── evaluate.py         # evaluate_state(state, your_index) -> float (Section 4.4)
    │   ├── policy.py           # obs -> SelectType dispatch -> action; the only thing main.py calls
    │   ├── budget.py           # wall-clock budgeting/cutoffs
    │   ├── logging_utils.py    # structured (state,action,outcome) logging — dev/harness only, stripped from shipped build
    │   └── config.py           # EVAL_WEIGHTS, PER_DECISION_BUDGET_SECONDS, num_determinizations — one importable dict
    ├── tests/                  # test_carddb.py, test_determinize.py, test_search_smoke.py, test_legality.py
    ├── decks/                  # candidate_v1.csv..v5.csv + deck_notes.md
    ├── harness/                # local_match.py, run_matches.py, results/
    ├── scripts/                # build_submission.sh, validate_submission.py, explore_carddb.py
    └── dist/                   # gitignored build output
```

Rules:
- **Never reimplement the ctypes bindings.** Vendor `sample_submission/sample_submission/cg/`
  verbatim; `build_submission.sh` always copies a fresh, unmodified copy into the packaged artifact
  at build time (even though dev uses a symlink for convenience).
- `main.py` stays a thin wrapper (mirrors the sample's structure) delegating to
  `policy.agent_decide(obs)` — keeps the Kaggle-facing entrypoint stable while internals iterate.
- Packaged layout (built by `build_submission.sh`) is flattened per Kaggle's requirement: top-level
  `main.py`, `deck.csv`, `cg/`, and one `ptcg_agent/` package directory (excludes `logging_utils.py`
  / dev-only code).
- Runtime code under `src/ptcg_agent/` must import stdlib + `cg` only — no `pandas`/`pytest` etc.

## 2. Card knowledge base (`carddb.py`)

Load once at import from `cg.api.all_card_data()`/`all_attack()` (not the CSV — CSV is for offline
human research/cross-checking only):

- `CARD_BY_ID: dict[int, CardData]`, `ATTACK_BY_ID: dict[int, Attack]`, `CARDS_BY_NAME`,
  `BASIC_ENERGY_IDS`, `ACE_SPEC_IDS` — precomputed once, pure/side-effect-free after load.
- `energy_cost_met(attack, attached) -> bool` — cheap feasibility filter for move-generation pruning
  and opponent-hand plausibility (not authoritative; the real engine remains ground truth).
- `compute_damage(attacker, attack, defender, tools, boosts) -> int` — first-order estimate (base
  damage, weakness, resistance); explicitly does **not** generically parse conditional attack text.
- `CARD_TEXT_DAMAGE_MODIFIERS: dict[int, Callable]` — small hand-curated override table for the
  ~10-20 highest-value cards relevant to our deck/common threats.

**Explicitly out of scope for Phase 1: a general attack/ability text parser.** With ~2,022
attack/ability rows of free text, this is a multi-week project on its own. Instead, lean on the real
search engine (`state.step()`) as ground truth for what a move *does* — the heuristic only needs to
score resulting states, not predict effect text. This is a permanent, documented limitation, not a
"TODO."

## 3. Deck selection

**Build a focused, single-energy-type efficient-attacker deck** rather than adapt the sample's
mono-Water mill deck. Rationale: a deck whose win condition is "attach energy, attack for
near-lethal, repeat" is far easier for a heuristic+shallow-search agent to pilot correctly than a
combo/mill deck requiring precise sequencing — and early on, our own play quality (not deck power)
is the bottleneck. Strong candidates already identified from the card pool: **Ceruledge** (220 dmg /
1 Fire energy) or **Palafin ex** (250 dmg / 1 Water energy after setup).

First placeholder decklist (Week 1): pick one such attacker line (3-4 copies per evolution stage),
add high-value competitive Supporters for consistency (Lillie's Determination, Judge, Boss's Orders,
Cyrano), 12-16 Basic Energy of the matching type (much less than the sample's 35, since cost per
attack is low), 1 ACE SPEC if it synergizes (e.g. Maximum Belt vs. ex-heavy opponents), verified
against the legality checker before ever calling `search_begin`/`battle_start`.

Treat this as a placeholder to unblock end-to-end testing, not final. Once the local harness exists
(Section 5), build 3-5 candidate decklists and round-robin them locally (same agent piloting each) to
get empirical relative-strength signal — ideally before the first submission, but must not block it;
`deck.csv` is cheap to swap later without touching agent code.

## 4. Core agent architecture

### 4.1 Top-level dispatch (`policy.py`)

```python
def agent_decide(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    try:
        return _decide(obs)
    except Exception:
        log_exception_once()
        return _fallback_decide(obs)   # trivial, cannot itself fail — e.g. first legal minCount options
```

`_decide` branches on `obs.select.type` only (never `SelectContext`):
- **Small/cheap selects** (YES_NO, COUNT, SPECIAL_CONDITION, single-legal-attack, EVOLVE,
  switch/retreat menus): resolved by direct heuristic rules or 1-ply `evaluate_state` lookahead —
  not worth a full determinization+search cycle.
- **MAIN-phase and trajectory-changing decisions**: routed through full PIMC search (`search.py`).
- **Large combinatorial CARD selects** (deck search): heuristically shortlist candidates first, then
  1-ply-score only the shortlist — never blindly full-search a 20+-option deck search.

### 4.2 Determinization sampling (`determinize.py`)

- **Own deck/prize**: not actually probabilistic — exact set-subtraction bookkeeping. Start from the
  known 60-card `deck.csv`, subtract everything observed moving via `obs.logs`; prize identity
  resolves via elimination as prizes get revealed. Implement as deterministic tracking, not sampling
  (a bug here would corrupt every `search_begin` call, so it needs dedicated unit tests).
- **Opponent deck/hand/prize/active**: genuine sampling via an `OpponentBelief` object, incrementally
  updated from `obs.logs` (known / constrained / fully-unknown per card). For fully-unknown slots,
  sample from an `ArchetypeConsistencyPrior` — weighted toward types/colors already observed in the
  opponent's play, never pure-uniform over all 1,267 cards (which would produce implausible worlds).
- **Legality gate**: every sampled determinization must independently pass the same legality checker
  from `carddb.py` before calling `search_begin` (bounded retries, then fall back to a generic legal
  filler deck) — a pathological belief state must never crash a turn's decision.
- **Budget**: N=6-10 determinizations per non-trivial decision, run strictly sequentially (per the
  shared-RNG-per-tree finding — no in-process parallel search). Conservative default total budget of
  **1.5-2.5s wall-clock per decision**, single tunable constant in `config.py`
  (`PER_DECISION_BUDGET_SECONDS`), enforced by `budget.py` checking `time.monotonic()` between every
  determinization and tree expansion, bailing to "best answer so far" on expiry. **This number is a
  placeholder pending empirical measurement against the real Kaggle per-move timeout — treat probing
  it as a top priority of the first real submission.**
- Cross-process parallelism (one OS process per determinization, doubling throughput on the 2 vCPUs)
  is a valid Phase 1.5 optimization, explicitly deferred — ship the safe sequential version first.

### 4.3 Search shape: shallow PIMC rollout with root-averaging (concrete default, not full ISMCTS)

Full UCT-tree ISMCTS needs many iterations to pay off — not affordable within a low-single-digit-second,
2-vCPU, ctypes-round-trip-per-step budget. Default instead to **root-parallelized PIMC**: for each
sampled determinization, do a shallow fixed-depth expectimax (2-3 of our decision points, opponent's
response approximated heuristically) via `search_begin`→repeated `search_step`, score leaves with
`evaluate_state`, then **average action scores across all N determinizations** at the root to pick
the final action. This is the concrete, resource-realistic version of "PIMC/ISMCTS-style search."

```python
def choose_action(obs, belief, budget, evaluator=evaluate_state) -> list[int]:
    ...  # sample determinizations, search_begin per one, enumerate_candidate_actions, rollout, average, select_best

def _rollout(search_id, action, depth, budget) -> float:
    ...  # apply action via search_step, greedy 1-ply continuation to `depth`, evaluate_state() at leaf/terminal
```

Implementation notes: always `search_release`/`search_end` every explored node (uncapped pool = leak
risk against the 12.2 GiB RAM cap); candidate-action enumeration reuses the same heuristic shortlist
logic as Section 4.1 for combinatorial menus, shared in one module rather than duplicated. Defer real
UCT/node-reuse ISMCTS to a later iteration if profiling shows headroom.

### 4.4 Heuristic evaluator (`evaluate.py`)

```python
def evaluate_state(state: State, your_index: int) -> float: ...
```

Weighted sum of small, independently-testable sub-scores (weights in `config.EVAL_WEIGHTS`, a flat
dict, for later autotuning):
- **Prize differential** (dominant term — the actual win condition)
- **Board HP swing** (your Pokémon HP% minus opponent's)
- **Active lethal-threat term** (can I be KO'd next turn / can I KO them — distinct from raw HP%)
- **Energy tempo** (attached energy count + attack-affordability via `energy_cost_met`)
- **Board development** (Pokémon count, evolution progress, bench fill)
- **Hand quality** (own hand size, playable draw/search Supporters, legal attack availability)
- **Special conditions** (penalty/bonus for status effects)
- **Terminal states** short-circuit to a large sentinel before any of the above

Each sub-score is its own named function (`_score_prizes`, `_score_hp`, etc.) for unit testing and
later autotuning.

## 5. Local test harness (`harness/`)

Built directly on `cg.game.battle_start`/`battle_select`/`battle_finish` — no Kaggle runner needed.

- `local_match.py::run_one_match(deck_a, agent_a, deck_b, agent_b, max_actions=3000) -> MatchResult`
  drives one game, always calling `battle_finish()` in `finally`. Captures winner, result/reason
  code, turn/action counts, wall-clock time, and any exception raised inside either agent (recorded
  as a forced loss + logged with the offending obs).
- **`Battle.battle_ptr` is a class-level singleton** — only one match per process. `run_matches.py`
  must use `multiprocessing`/`ProcessPoolExecutor` (one match per worker process), never in-process
  concurrent matches.
- Match suites to run: (a) candidate vs. sample's random agent (sanity baseline), (b) candidate vs.
  sample's mill deck, (c) candidate vs. itself (determinism/self-consistency), (d) round-robin across
  3-5 candidate decklists. Collect win rate, game length, and **fallback-path trigger rate** (a proxy
  for hidden bugs) into CSVs under `harness/results/`.
- Structure the match-driving loop so it's directly reusable later as the skeleton of a self-play
  data-generation harness (only the per-decision logging payload changes, not the driving mechanics).

## 6. Submission packaging & validation

`scripts/build_submission.sh`:
1. Wipe/recreate `dist/staging/`.
2. Copy `src/ptcg_agent/*.py` (excluding `logging_utils.py`/dev-only code) → `staging/ptcg_agent/`.
3. Copy `main.py` → `staging/main.py` (top level, per Kaggle's requirement).
4. Copy the chosen `decks/candidate_vN.csv` → `staging/deck.csv`.
5. Copy `sample_submission/sample_submission/cg/` fresh (all 4 compiled binaries) → `staging/cg/`.
6. `tar -czf dist/submission.tar.gz -C staging .` — verify `main.py` is at the tar's top level.
7. Check size against the 197.7 MiB cap (currently trivial: ~5.3 MB of binaries).
8. **Hard-fail the build** (not just warn) if the deck fails the legality checker.

`scripts/validate_submission.py` — smoke test the **packaged artifact** (subprocess without dev
`PYTHONPATH`):
- Runs standalone with only stdlib + bundled `cg`.
- `read_deck_csv()` works from cwd and via the `/kaggle_simulations/agent/` fallback path.
- Runs at least one full local match via the packaged `main.py`, zero uncaught exceptions.
- Explicitly exercises edge cases: `minCount == 0`, forced full-hand discard, ability/skill ordering,
  `COUNT` selects, the initial `obs.select is None` deck request — hand-construct matchups to hit
  these if incidental coverage doesn't.
- Confirms the `try/except` fallback wrapper is present in the packaged build's actual call path, and
  that `_fallback_decide` itself cannot fail (no dependency on search/carddb/network/disk).
- No absolute paths, no disk writes in the shipped build.
- Times a sample of decisions from the packaged artifact against `PER_DECISION_BUDGET_SECONDS` as a
  proxy check pending the real Kaggle-side timeout measurement.

## 7. Hooks for the later learned-evaluator phase

- **`evaluate_state(state, your_index) -> float`** is the seam — `search.py` should accept an
  `evaluator` parameter (defaulting to the heuristic) so Phase 2 swaps in a neural net forward pass
  without touching search control flow.
- **Structured per-decision logging** (`logging_utils.py`, active only in harness runs, stripped from
  shipped builds): log `{state_features, action_taken, determinization_summary, chosen_score,
  eventual_outcome}` per decision to JSONL — exactly the (state, action, outcome) tuples needed to
  bootstrap a value net later, versioned via a `schema_version` field.
- **`config.EVAL_WEIGHTS`** as a flat importable dict is deliberately shaped for an
  autoresearch-style overnight autotuning loop later: perturb weights, run a fixed budget of local
  matches via `run_matches.py` against a fixed opponent pool, score by win rate, keep/discard. Note
  this needs no GPU — the school cluster's value here is CPU core count for parallel match
  simulation, not tensor compute, until an actual learned net exists.
- **`OpponentBelief`**/`ArchetypeConsistencyPrior` is a natural seam for a future learned
  opponent-modeling component, without touching `search.py`.

## 8. Week-by-week timeline (compressed)

Priority throughout: get something legal and non-crashing on the ladder **early**, then iterate.
Compressed from the original 5-week draft to buy a full extra week for the self-play RL track,
since a search-only agent is unlikely to be sufficient for a top-10-of-5,000 finish on its own.

- **Week 1 (Jul 13-19)**: scaffolding, vendor `cg`, `carddb.py` + tests, legality checker, first
  placeholder decklist, ship a **heuristic-only (no search)** agent — legal, crash-resistant,
  packaged and validated per Section 6. **Submit by day 3-4 of the week** (not day 7) specifically
  to start the clock on probing the undocumented per-move timeout — the ladder clock only starts
  once something is actually submitted. Build the local harness and A/B test 3-5 candidate decklists
  while that submission accumulates games in the background.
- **Week 2 (Jul 20-26)**: build `determinize.py` + `search.py` (PIMC rollout + root-averaging);
  integrate into `policy.py`, keeping the Week-1 path as fallback/fast-path; tune budget constants
  using Week 1's empirical timeout signal; confirm search-augmented agent beats the heuristic-only
  agent locally with clear margin before trusting it.
- **Week 3 (Jul 27-Aug 2)**: submit the search-augmented agent, harden edge cases, finalize decklist
  from local A/B + real ladder signal — **locked in a full week ahead of the Aug 9 entry deadline.**
  In parallel, kick off the RL track: build the self-play data-generation harness (reusing the local
  match driver from Section 5) and the training loop skeleton on the school GPU cluster.
- **Week 4 (Aug 3-9, entry/merger deadline)**: full week dedicated to the RL track — train a learned
  value net via self-play to swap into `evaluate_state` (search/policy logic untouched), apply
  anti-cycling safeguards from the start (frozen-snapshot/opponent-pool style, per the Lux AI/NFSP
  findings — do not run naive vanilla self-play), continuously A/B against the Week 3 locked-in
  search agent in local testing.
- **Week 5 (Aug 10-16, final deadline)**: decision checkpoint around **Aug 14** — ship the RL version
  only if it's clearly and consistently beating the locked-in search-only agent in local testing;
  otherwise ship the proven search-only agent, which is never at risk since it's already locked in.
  Reserve the last 1-2 days purely for validation — no risky last-minute changes.
- **Aug 17-31**: no further submissions possible. Use remaining GPU cluster time to continue the RL
  track for a future iteration if it didn't make the cutoff, and log postmortem learnings (which
  heuristic terms mattered, whether search vs no-search moved the rating, whether the RL version
  would have won the A/B given more time) regardless of what shipped.

## Explicit risks and how the plan handles each

1. **Undocumented per-move timeout** — conservative default budget + empirical probing via an early
   real submission + fast heuristic-only fallback path.
2. **RNG shared across one search tree** — strictly sequential traversal within a tree; any future
   parallelism is cross-process only, never in-process threads sharing one `agent_ptr`.
3. **`SelectContext` unreliable** — all control flow keys off `SelectType`/`Option.type` only.
4. **No `SearchShuffle` binding exposed to Python** — likely fine since each decision re-samples a
   fresh determinization anyway; flagged for early smoke-test validation, not independently fixable
   (no build toolchain for the compiled binaries is present in this repo).
5. **Deck chosen under zero empirical strength data initially** — explicit local A/B testing plan
   before treating any deck as final; `deck.csv` is cheap to swap later.
6. **No general attack/ability text parser** — permanent, documented Phase 1 limitation; real search
   engine resolution is the source of truth, not text prediction.
7. **Own-prize/deck bookkeeping bugs would corrupt every `search_begin` call** — implemented as exact
   deterministic tracking (not sampling) with dedicated unit tests.
8. **`Battle` singleton limits local harness to one match per process** — `run_matches.py` uses
   multiprocessing, documented in `harness/README`.
9. **macOS dev machine vs. Kaggle's Linux inference container** — package ships all 4 provided
   binaries regardless of dev platform; `sim.py`'s own platform dispatch picks the right one at
   runtime; smoke-test on a Linux environment before first submission if available.
10. **Self-play RL track might not converge, or might quietly cycle into a bad strategy, within the
    compressed Week 3-5 window.** Handled by: baking in known anti-cycling safeguards
    (frozen-snapshot/opponent-pool distillation, per the Lux AI Season 1 precedent) from the start
    rather than attempting naive self-play first; treating the Week 3 locked-in search-only agent as
    a guaranteed, never-at-risk fallback; and gating the swap on a hard local-A/B-win requirement at
    the Aug 14 decision checkpoint rather than shipping the RL version on faith.

## Critical files

- `sample_submission/sample_submission/cg/api.py` — the entire `search_begin`/`search_step`/
  `all_card_data` surface the architecture is built on; vendor unmodified.
- `sample_submission/sample_submission/cg/game.py`, `cg/sim.py` — local harness foundation;
  `sim.py`'s singleton `Battle` class dictates the multiprocessing requirement.
- `EN_Card_Data.csv` — offline research/cross-check reference only, not runtime source of truth.
- `ptcg_engine/ptcgProgram 22/Search.h` — ground truth for search-tree semantics (shallow `alloc()`,
  shared `Game*`/RNG, auto-advance loop) that directly shaped the sequential-search and
  result-parsing design above.
- `sample_submission/sample_submission/main.py`, `deck.csv` — structural template for `main.py` and
  the exact `deck.csv` format.

## Verification

- Unit tests (`tests/`) for `carddb.py` lookups, deck legality, and `determinize.py`'s own-state
  bookkeeping, run via `pytest`.
- Local harness matches (Section 5) as the primary end-to-end signal: agent-vs-sample-random,
  agent-vs-sample-mill, agent-vs-self, and cross-decklist round robins, tracked by win rate/game
  length/fallback-trigger rate.
- `scripts/validate_submission.py` run against the actual packaged `.tar.gz` before every submission
  (Section 6 checklist) — this is the closest available proxy for Kaggle's real runtime before an
  actual submission confirms it.
- First real Kaggle submission (target: Week 2) doubles as validation of the packaging pipeline and
  as the only way to empirically learn the true per-move timeout.

## Notes and clarifications

**Plan in one paragraph:** we can't see the opponent's cards, but we do have the real game engine in
our pocket. So instead of guessing once, we make several different reasonable guesses about what the
opponent might be holding, replay each guess forward in the real engine to see how a move plays out,
and average the results to decide what to actually do — that's the whole trick. On top of that
guessing-and-searching agent, we're using a simple hand-built scorecard (prizes taken, health, board
state, etc.) instead of a trained model at first, because a trained self-play model is a multi-week
project that can quietly go wrong (a real, documented failure mode) if rushed — so we ship the safer
version first and only swap in a learned model later if it demonstrably wins more.

**What "ply" means:** one single move by one single player, not a full turn. A 1-ply lookahead means:
try each option, imagine the very next position after making it, score just that position, and stop —
no further guessing about what happens after. It's the cheap version of lookahead, used for small/easy
decisions. A 2-ply lookahead would also imagine the opponent's best response before scoring, and so
on, each additional ply going one move deeper into the "if I do this, then they do that" chain.

**What "PIMC" means (Perfect Information Monte Carlo):** our chosen search method. Normal game-tree
search assumes you can see the whole board with no secrets. Our game has secrets (the opponent's
hand). PIMC's trick: pretend the secret doesn't exist, one guess at a time. Guess a plausible
opponent hand, treat that guess as if it were simply true (no hidden information left — "perfect
information"), and run ordinary search on that pretend-perfect-information version of the game.
Repeat with a handful of different guesses, then average the results across all of them to pick the
actual move. It's the simpler, cheaper cousin of `ISMCTS`, chosen specifically because it fits the
2-vCPU inference budget, whereas the fancier method needs more compute to pay off.
