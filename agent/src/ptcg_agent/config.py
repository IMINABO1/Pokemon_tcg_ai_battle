"""Central tunable constants. Flat dicts / simple scalars, shaped for later autotuning."""

# Per-decision wall-clock budget (seconds).
# EMPIRICAL (2026-07-20, episode 87163505): the engine enforces NO per-move timeout
# (actTimeout=0). The binding constraint is runTimeout=2000s for the WHOLE episode of
# total agent compute (episodeSteps effectively unlimited at 10,000,000). So the budget
# must be managed against the remaining episode-time pool, not a fixed per-move cap.
# 2.0s/decision is a safe default: even a pathological ~1500-decision game stays well
# under 2000s. budget.py should scale this down adaptively as episode time is consumed.
PER_DECISION_BUDGET_SECONDS = 2.0

# Whole-episode compute budget (seconds) — the real hard limit (Kaggle runTimeout).
EPISODE_TIME_BUDGET_SECONDS = 2000.0
# Safety margin: stop searching once this fraction of the episode budget is spent.
EPISODE_TIME_SAFETY_FRACTION = 0.85

# Number of opponent-hand determinizations per non-trivial decision (Week 2+, search phase).
NUM_DETERMINIZATIONS = 8

# Heuristic evaluator weights (flat dict for later autotuning). Used in Week 2+.
EVAL_WEIGHTS = {
    "prize_diff": 100.0,       # dominant term: the actual win condition
    "board_hp_swing": 1.0,
    "lethal_threat": 20.0,
    "energy_tempo": 3.0,
    "board_development": 5.0,
    "hand_quality": 2.0,
    "special_condition": 8.0,
}

# Large sentinel for terminal states in evaluate_state.
TERMINAL_WIN = 1e9
TERMINAL_LOSS = -1e9
