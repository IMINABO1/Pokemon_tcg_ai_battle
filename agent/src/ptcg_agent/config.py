"""Configuration constants and evaluation weights for PTCG Agent."""

PER_DECISION_BUDGET_SECONDS: float = 2.0
NUM_DETERMINIZATIONS: int = 6
MAX_SEARCH_DEPTH: int = 3
MAX_ACTION_CANDIDATES: int = 8

EVAL_WEIGHTS = {
    # Prize differential: dominant term (your prizes taken - opponent's prizes taken)
    "prize_diff": 100.0,
    "own_prizes_remaining": 0.0,
    # Active lethal-threat term (can KO active / active is lethal)
    "can_ko_active": 40.0,
    "active_in_danger": -35.0,
    # Board HP swing (your HP sum - opponent HP sum)
    "hp_swing": 0.4,
    # Energy tempo: attack-affordability matters more than raw attached count
    "energy_attached": 1.5,
    "attack_ready": 30.0,
    # Board development: bench fill, evolution status
    "bench_count": 5.0,
    "evolution_advantage": 15.0,
    # Hand quality: mild card-advantage only
    "hand_size": 0.5,
    # Status condition penalties (own active)
    "status_poisoned": -5.0,
    "status_burned": -5.0,
    "status_asleep": -15.0,
    "status_paralyzed": -25.0,
    "status_confused": -10.0,
}
