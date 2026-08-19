# Problems Encountered

## 2026-08-01 — PIMC agent loses to a random agent (~25% win rate) — and the first root-cause guess was wrong

**Problem** — The complete PIMC + heuristic agent wins only ~1 in 4 local matches against a
purely random baseline (5/20 in the measured suite). A forward-search agent losing to random is
not variance or weak tuning — something structural is off.

**(Investigation)** — Two rounds, and the second overturned the first:

1. *Code-read hypothesis (wrong):* reading the source suggested the search **never actually runs** —
   every failure path (`search.py` per-candidate `except: pass`, per-determinization
   `except: continue`, `policy.py` whole-decision fallback) swallows exceptions with no counter, and
   when all rollouts fail the code returns `candidates[0]`, i.e. a deterministic "always option 0"
   policy. The apparent evidence was old suite timing (matches finishing in ~0.6–2 s, seemingly too
   fast to be searching).

2. *Instrumentation (correct):* rather than act on the guess, we added counters at every swallow
   site (`agent/src/ptcg_agent/stats.py`) and surfaced them into the results CSV. Over 20 matches:
   **0 whole-decision fallbacks, 0 `search_begin` failures, 17,532 / 17,532 candidate rollouts
   scored, 0 rollout exceptions.** The search runs fully on every decision. A follow-up probe showed
   it also *discriminates*: on 629 search decisions, 23.8% picked a non-default action and only
   23.7% were degenerate ties (all candidates scoring equal). So the "always option 0" hypothesis is
   **refuted** — the scoring is real and mostly non-trivial.

   The true signal: **win/loss tracks game length.** Every PIMC win was a short game (15–36 actions);
   every loss was long (148–192). The agent gets weaker the longer a game runs.

**Resolution** — Re-point Phase 0 away from the (non-existent) plumbing failure and toward
**decision quality**: (a) the evaluator — ~24% of decisions are ties it can't break, and the
weight scheme rewards hoarding (`supporter_in_hand +8`, `hand_size +2` make playing a supporter
cost −10) and undervalues damage (`hp_swing 0.1` vs `energy_attached 10`); (b) rollout depth (3
prompts < one turn) and the greedy opponent continuation; (c) opponent-belief drift over long games
(`update_from_logs` re-scans and re-appends every decision without dedupe — plausibly why late-game
play degrades). The instrumentation counters (fallback rate, degenerate-tie rate) stay in as the
regression signal for each fix. Status ⏳ open (baseline 25% vs Random; gate G0 = ≥90%).

**Why this over alternatives** — We nearly spent the first days rewriting the rollout state
lifecycle (fresh `search_begin` per candidate, calling `search_release`) to fix a failure that
*was not happening*. The counters cost ~30 minutes and redirected the whole effort. The deeper
lesson — and the reason observability was made Phase 0 step 0, not the first fix — is that
**blanket exception-swallowing with no counter made a healthy search and a dead one look identical
from the outside; the only way to tell them apart was to measure.** A plausible code-read is a
hypothesis, not a diagnosis.

**Status** — ✅ resolved 2026-08-15 by the overhaul: all three suspected causes were real and fixed
(evaluator rebalance + turn-horizon rollouts; serial-deduped beliefs). Gate exceeded: 29/32 (90.6%)
vs Random. The "wins short / loses long" signal was the rollout parity bug (below) plus belief drift.

## 2026-08-15 — Rollout parity: fixed-depth rollouts punished attacking

**Problem** — The search agent chose RETREAT/END over available ATTACKs (42 skipped attacks in one
traced game) and lost long games it should have closed out.

**Investigation** — Traced MAIN-phase choices in a losing game: with options {ATTACK, RETREAT, END}
the agent retreated. Cause: rollouts ran a fixed number of *decisions* (depth 4). Attacking ends the
turn, so attack lines were evaluated *after the opponent's reply*, while passive lines burned all
plies on our own turn and were evaluated with the opponent frozen — comparing leaves at different
game phases made stalling always look better.

**Resolution** — Rollouts run to a fixed *turn horizon* (start of our next turn, `turn >= root+2`)
with a static develop-then-attack policy, so every candidate line is scored at the same phase.
vs greedy went 11/16 → 14/16 immediately.

**Why this over alternatives** — Deeper fixed depth has the same parity bug, just slower. Full
expectimax is impossible in-process (shared RNG per tree). Evaluating only at own-turn boundaries
was the cheapest phase-normalization available.

**Status** — ✅ resolved (commit `a5a8b97`)

## 2026-08-15 — Infernal Slash: the deck's main attack did nothing, all game, every game

**Problem** — With the Ceruledge deck the agent attacked every turn and never dealt damage; games
ended with 0–1 prizes taken.

**Investigation** — Board traces showed the opponent's active at full HP after a dozen of our
attacks. Card text: Infernal Slash (220) *"Discard 4 Basic Fire Energy cards from your hand. If you
can't, this attack does nothing."* The agent held <4 Fire (and the evaluator's uncapped
`energy_attached` term actively taught it to dump hand Fire onto the board — one game ended with 7
energies attached to a 1-energy attacker).

**Resolution** — Fire-aware evaluator terms (held Fire scored as ammo, KO/attack-ready gated on the
hand cost, attached-energy credit capped at 2/Pokémon) recovered only to ~50% vs random — so the
deck itself was retired for the sample shell, where the searcher immediately scored 16/16 vs random
and greedy. Root lesson: the engine prices attack riders correctly inside rollouts, so the searcher
is deck-agnostic — deck choice should follow searcher performance, not hand analysis.

**Why this over alternatives** — Teaching the evaluator every rider is unbounded work; switching to
a deck whose attacks have no hand-cost riders removed the whole problem class.

**Status** — ✅ resolved (deck retired; terms kept but inert)

## 2026-08-15 — Determinization gate could never pass; beliefs grew without bound

**Problem** — Every determinization burned all legality retries; opponent worlds were built from an
ever-growing list of "seen" cards.

**Investigation** — The gate called full-deck `is_legal()` (requires exactly 60 cards) on hidden
zones only — impossible once the opponent has cards in play, so the gate was dead code. Separately,
`update_from_logs` appended `cardId` per log entry; the same physical card is re-logged on every
move, so the belief pool inflated indefinitely (Belex's 08-01 entry had flagged this as the likely
long-game degrader).

**Resolution** — Partial-legality check (per-name copy caps over hidden + visible cards, no
60-card requirement); beliefs deduped by card `serial`; opponent pool = mirror archetype prior
minus visible cards.

**Status** — ✅ resolved (commit `a5a8b97`)

## 2026-08-15 — The real timing model: actTimeout=0 with a 600 s per-agent bank

**Problem** — PLAN v1's "biggest open unknown": the per-move timeout was undocumented, and the
configured 2.0 s/decision budget was a guess.

**Investigation** — The week-1 episode JSON's specification block documents it:
`actTimeout: 0`, per-agent `remainingOverageTime` bank of 600 s (`shared: false`), DQ at 0. A fixed
2.0 s/decision would DQ ~300 decisions into a long game. The value lives only in the raw obs dict —
`to_observation_class` silently drops unknown keys.

**Resolution** — Read the bank before dataclass conversion; adaptive budget
clamp(bank / expected-remaining-decisions, 0.3–2.5 s); search skipped entirely under 40 s.
Ladder telemetry: worst real game used ~62 s of the bank.

**Status** — ✅ resolved (commit `a5a8b97`)

## 2026-08-16 — Kaggle runner exec()s main.py: no __file__, last callable wins

**Problem** — v0.2 went `SubmissionStatus.ERROR` on the ladder despite passing every local check.

**Investigation** — Installed `kaggle-environments` and ran the actual `cabt` env: the runner
`exec()`s the agent file with an empty namespace — `__file__` is undefined (our `main.py` resolved
`sys.path` via `Path(__file__)` at module level → instant NameError) — and takes the *last
callable defined* in the file as the agent. Local `import main` can never catch either behavior.

**Resolution** — Guarded bootstrap (try `__file__`, fall back to `/kaggle_simulations/agent` and
cwd); `agent()` kept as the file's final definition; a real cabt episode added to the pre-submission
checklist. Resubmission validated instantly.

**Why this over alternatives** — Reproducing the exact production harness locally beat guessing from
docs; the failing layer was the harness contract, not the agent.

**Status** — ✅ resolved (commit `a7dbd34`)

## 2026-08-16 — Benchless insta-losses: the deck, not the search, created dead positions

**Problem** — v0.2.1 went 5-8 (~478) on the ladder with no crashes or timeouts; one loss ended on
turn 3 with all six prizes intact.

**Investigation** — Replayed a real ladder decision through the agent with score instrumentation:
ATTACK scored −1,000,000 in 8 of 16 sampled worlds — correctly, because after any exchange the
opponent's counter-KO hit our benchless board (instant loss). With only 8 basics the deck frequently
never benched; the search was accurately evaluating positions the deck made unwinnable. Meta
opponents run 4× Buddy-Buddy Poffin with ≤70 HP basics; every basic in our shell exceeds Poffin's
limit.

**Resolution** — Deck v4: 13 basics (+3 Suicune, +2 Glastrier) plus a benchless evaluator penalty
(insurance beyond the rollout horizon). Ladder ~570. Superseded by the mined top-meta list (v0.4b,
~760): archetype was the binding variable.

**Status** — ✅ resolved (commits `3f8468f`, `4b31215`)

## 2026-08-16 — Replay schema: actions answer the previous step, and yourIndex can't identify the actor

**Problem** — First parse of real ladder replays produced decisions with fewer choices than
`minCount` (impossible actions) and empty deck lists.

**Investigation** — kaggle-environments pairing: the action recorded at `steps[t]` was produced from
the observation at `steps[t-1]` (deck lists live in step 1's actions answering step 0's null obs).
And `yourIndex` is self-relative in *every* agent's observation — both players see themselves as
"you" — so it cannot identify the actor; INACTIVE agents echo a stale obs alongside an empty action,
which is indistinguishable from a legal "pass" by content alone.

**Resolution** — Filter on `steps[t-1][p].status == "ACTIVE"`, pair obs(t−1) with action(t).
Validation: 806 decisions across 5 episodes, 0 minCount violations, leakage audit exit 0.

**Status** — ✅ resolved (commit `8ce4e6a`)

## 2026-08-16 — Oversubscribed benchmark rig produced fake 0-8 timeout losses

**Problem** — In the Elo-band gauntlet, all three candidate agents went 0-8 against the band-900
Lucario deck with every game ending "exhausted overage bank (TIMEOUT)" after only 10–20 decisions
(~40 s/decision wall time), suggesting a catastrophic deck-specific engine hang.

**Investigation** — A single-game probe with per-call timing against the same deck showed every
decision ≤1.3 s and every engine call fast — no hang exists. The suite had been run as 3 concurrent
jobs × 5 workers = 15 simultaneous search agents on 16 cores *alongside Docker Desktop's VM*;
memory/CPU contention inflated wall-clock ~30×, and the harness's simulated 600 s bank correctly
charged the inflated time as timeouts. The "deck interaction" was the test rig.

**Resolution** — Rerun the full gauntlet sequentially at the proven 8-worker configuration. Rule
recorded: wall-clock-budgeted benchmarks must never share the machine with other heavy jobs; scale
parallelism by *games per job*, never by concurrent jobs.

**Why this over alternatives** — Trusting the 0-8 would have disqualified strong candidates on
corrupted data the night of the deadline; hunting a phantom engine hang would have burned the
remaining hours.

**Status** — ✅ resolved (gauntlet rerunning clean)

## 2026-08-17 — Greedy-pilot gauntlets did not transfer to the real ladder (v0.5 collapse)

**Problem** — v0.5 (Dragapult, the 1000-band's dominant deck) scored 22/24 in local gauntlets
against the band-winning decks, then went 37-56 (Elo 412) in real ranked play — losing to
opponents rated 300-500. The final-slot decision was made on evidence that did not transfer.

**Investigation** — Two rounds; the second (replay forensics, 14 of our games vs 60 top-band
Dragapult player-games) overturned the first. (a) *Proxy gap (confirmed):* gauntlet opponents were
greedy scripts; real opponents are competent rule-based agents, so measured edges were inflated.
(b) *First skill-gap hypothesis (REFUTED):* "our search can't do combinatorial counter placement" —
false; placement prompts are trivial 1-of-N selects and we answer them all. The counters simply
never get generated: winners fire **Phantom Dive 3.8×/game** (≈27 placement prompts/game,
concentrated targets, 4.3 prizes/game, 0.83 prizes/attack); ours fires it **once per ~2.6 games**
and attacks with **Jet Headbutt (70, any 1 energy, no effect)** instead — 1.5 prizes/game. Root
cause in code: Phantom Dive costs {R}{P} *specifically*, and the evaluator's `attack_ready` asks
"is ANY attack affordable?" — Jet Headbutt satisfies it with one energy of any type, so attaching
the second, type-correct energy earns nothing and the two-turn assembly plan is invisible. Same
failure class as Infernal Slash (attack-enabling precondition a generic evaluator can't see); the
Ceruledge fix was never generalized to "value the strongest affordable attack, not any". The
greedy-pilot gauntlet masked it: 70-damage beatdown beats greedy scripts anyway.

**Resolution** — None possible post-deadline; logged as the campaign's decisive mistake. For any
future entry: (1) benchmark against the strongest available *pilots* (public rule-based agents),
not just strong decks under weak pilots; (2) audit whether a candidate deck's win condition flows
through a decision type the searcher genuinely optimizes; (3) treat "necessary but not sufficient"
evidence labels as binding at decision time, not as footnotes.

**Why this over alternatives** — Submitting v6 Lucario (21/24 locally, lower skill floor, its
attacks need no combinatorial targeting) would likely have out-scored v0.5; it was ranked below
v0.5 on exactly the evidence that failed to transfer.

**Status** — ❌ unrecoverable (final-slot outcome); lesson recorded for the write-up

## 2026-08-16 — Branch merge silently mutated validated agent code

**Problem** — After merging `origin/imi` (to push), the working tree's evaluator weights and search
code no longer matched what was benchmarked and live on the ladder — despite merging with `-X ours`.

**Investigation** — `-X ours` only resolves *conflicting* hunks in our favor; the other branch's
non-conflicting hunks (an 08-01 evaluator rebalance written against the pre-overhaul agent) applied
cleanly. Git also silently overwrites *ignored* files with tracked versions from the merged branch —
which is how the gitignored worklogs were clobbered by their tracked counterparts.

**Resolution** — Restored the five affected files to the validated revision; kept additive files
(`stats.py`, `test_evaluate.py` with its invariant test xfail'd as a documented divergence); merged
both worklog histories into single tracked files.

**Why this over alternatives** — Keeping the merged mutations would have put unvalidated behavior on
the branch that claims to be what's on the ladder; reverting the whole merge would have lost the
teammate's history and blocked the push.

**Status** — ✅ resolved (commit `2e4fbf3`)
