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

**Status** — ⏳ open (until gate G0: ≥90% vs Random over ≥100 matches)
