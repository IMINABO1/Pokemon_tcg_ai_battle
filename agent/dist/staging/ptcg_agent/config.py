"""Central tunable constants. Flat dicts / simple scalars, shaped for later autotuning."""

# Per-decision wall-clock budget (seconds). PLACEHOLDER pending empirical measurement
# of the real Kaggle per-move timeout (top priority of first real submission).
PER_DECISION_BUDGET_SECONDS = 2.0

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
