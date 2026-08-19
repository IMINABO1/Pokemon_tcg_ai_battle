# The Ladder Is Stratified: Evidence-Driven Deck Selection for a Determinized-Search Agent

*Draft v1 — target ≤2,000 words. Team: Prof Oak's lab.*

## Summary

We built one agent and changed only its deck. The agent is a determinized
search (PIMC) over the competition's real C++ engine: at every decision it
samples completions of hidden information, steps each candidate action
forward through the actual rules engine to a fixed *turn* horizon, and picks
the action with the best average evaluation. Around that fixed pilot we ran
an evidence loop on 380+ mined public replays: diagnose our losses from
ranked games, extract the decks that beat us, measure the meta band by band,
and refit the deck. Three findings drove every decision, and we believe they
generalize beyond our entry:

1. **The engine prices the cards, so search is deck-agnostic** — attack
   riders, coin flips, and abilities cost nothing to model because rollouts
   run in the real engine. Swapping decks under the same pilot moved us more
   than any code change.
2. **The ladder is stratified by archetype, and each deck carries a visible
   ceiling.** From 267 sampled replays across six Elo bands: Grimmsnarl
   dominates 650–850 (up to 68% of players, ~67% of wins) and wins nothing at
   900; Mega Lucario owns 900 (48% share, 57% of wins) and vanishes at 1000;
   Dragapult ex alone rules 1000+ (64% share, 70% of wins). Because a
   submission's deck is frozen, **deck choice is ceiling choice**.
3. **Play-quality gaps are measurable from replays.** Winners at every band
   average 3.1–3.8 benched Pokémon and attack by turn ~4; our early agent
   benched 1.9 and was benchless on 10% of its turns — a property we found,
   fixed, and re-measured from ranked-game evidence.

## The agent

**Search.** `search_begin/step/end` exposes a true clone-and-step forward
model. Per decision we sample N=32 determinizations of hidden zones, roll
each candidate action forward with a static develop-then-attack policy for
both players, stop at the start of our next turn, and average a hand-tuned
evaluation (prizes ≫ lethal threats ≫ tempo/board terms) across worlds.

**Two design choices mattered most.**
- *Turn-horizon rollouts, not fixed depth.* Fixed-depth rollouts compare
  "attacked → opponent replied" leaves against "stalled → nobody replied"
  leaves at different game phases; our agent literally stopped attacking
  (42 skipped attacks in one traced game). Evaluating every line at the same
  phase (start of our next turn) fixed it: 11/16 → 14/16 vs a greedy
  baseline from this change alone.
- *Determinization discipline.* Beliefs dedup revealed cards by physical-card
  serial (they re-log on every move; the un-deduped pool grows without
  bound), opponent worlds are a mirror-archetype prior minus all visible
  cards under per-name copy caps, and a partial-legality gate validates
  hidden zones plus visible cards (a full-deck check can never pass
  mid-game).

**Timing.** The real constraint is not per-move: `actTimeout=0` with a
per-agent 600 s bank per game, disqualification at zero. We read the bank
from the raw observation each turn and budget
clamp(bank / expected-remaining-decisions, 0.3–2.5 s), with a search-free
fallback under 40 s. Measured worst case on the ladder: ~92 s of 600 used.

## The evidence loop (what we'd tell every future entrant)

**Losses are data.** Our first search submission sat at ~478. Replays showed
no crashes and no timeouts — it died *benchless*: with 8 basics it often
never developed a bench, and one KO ended the game. The search was scoring
these positions correctly (−10⁶ in half its sampled worlds); the deck was
dealing dead hands. Deck surgery (+basics) beat evaluator surgery.

**Opponents are teachers.** We walked the episode graph
(opponents-of-opponents, three hops) to the ~1280-rated teams and extracted
their exact 60-card lists from replays. Multiple top teams ran an identical
Grimmsnarl/Munkidori list; under our unchanged pilot it went 16/16 vs our
own previous deck and became our main entry (settled ~700–760, top ~20%).

**Census before commitment.** Before spending our final submission we
sampled ~50 replays from each band (600s–1000s) and measured archetype
share, winner share, and play metrics. The stratification result above fell
out immediately — including the answer to "would Dragapult fail down low?"
No: sub-800 Dragapult players hold ~50% with near-top-band play metrics;
they are in transit, not countered. Their climb path (650–850) is
wall-to-wall Grimmsnarl, Dragapult's best matchup.

**Local gauntlets over intuition.** Candidates played greedy pilots of each
band's winningest deck: our Grimmsnarl list swept 24/24; the 1000-band
Dragapult list went 22/24; the 900-band Lucario list 21/24 with its losses
concentrated against Dragapult — reproducing the ladder's own food chain in
vitro. Final portfolio: Grimmsnarl (floor, proven) + Dragapult (ceiling bet,
climbing at time of writing), chosen because ranking takes a team's best
entry, so entries should maximize *ceiling diversity*, not average strength.

**Measurement rigor is a survival skill.** Three of our worst bugs were
invisible without instrumentation: the runner `exec()`s `main.py` with no
`__file__` and takes the last callable defined (all local imports pass; the
ladder errors instantly — reproduce with the real `cabt` environment before
submitting); an oversubscribed benchmark rig (15 concurrent agents + a VM on
16 cores) faked 0-8 "timeout losses" that a single-game probe disproved in
minutes; and a code-read diagnosis of "search never runs" was refuted by
counters at every exception site. Measure, don't infer.

## Imitation learning at field scale (FIELD-Zero)

In parallel we built a replay-mining → behavior-cloning pipeline with a
strict leakage rule: a feature is a legal model input iff our own agent
could compute it at that decision from its own observation stream; the
opponent's end-revealed deck/archetype are auxiliary targets only, enforced
by an audit that fails the build. Schema quirks that cost us a day, frozen
from real replays: actions at step *t* answer observations at *t−1*; the
prompted agent is the one whose *status* is ACTIVE (`yourIndex` is
self-relative in every observation and cannot identify the actor); beliefs
must dedup by card serial. On the university cluster (Docker → Apptainer,
environment in the image, code via git) the pipeline mined and parsed
**69,542 games (~25M decisions)**; the dataset builder had to be rewritten
to stream (the naïve in-memory build exceeded 260 GB RAM). Behavior-cloning
results and a learned-evaluator vs hand-tuned-evaluator A/B inside the
search will be reported here. [PENDING: training curves, val_top1, A/B]

## Results

| Entry | Deck | Outcome |
|---|---|---|
| Week-1 heuristic | placeholder mono-Water | 344 |
| PIMC + Suicune shell | 13-basics Water | ~534–570 |
| PIMC + Grimmsnarl (mined) | top-meta list | ~694–758, top ~15–23% |
| PIMC + Dragapult (mined) | 1000-band list | climbing at time of writing [PENDING: final] |

Identical search code across the last three rows: the deck, chosen from
measured evidence, was the decisive variable at every step.

## What we'd do differently

Deck selection should have been evidence-driven from day one — we spent the
first weeks tuning a pilot around a deck whose signature attack ("discard 4
Fire from hand, else nothing") our evaluator couldn't see. And every local
benchmark should have been run against the production harness from the
start; the ladder's runtime contract, not the game rules, produced our only
outright submission failure.

---
*Word count target check: ~1,150 words — room for RL results, final ladder
numbers, and one figure (band × archetype share heatmap) before the 2,000
cap. Reproducibility artifacts: public GitHub repo (agent, mining pipeline,
Slurm scripts, benchmark harness, this journal).*
