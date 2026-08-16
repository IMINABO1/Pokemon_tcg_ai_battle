"""Configuration constants and evaluation weights for PTCG Agent."""
import os

PER_DECISION_BUDGET_SECONDS: float = float(os.environ.get("PTCG_BUDGET", "1.2"))
NUM_DETERMINIZATIONS: int = int(os.environ.get("PTCG_DETS", "32"))
# Rollouts run to a fixed TURN horizon (start of our next turn), not a fixed
# decision depth: comparing "attacked, opponent replied" leaves against
# "developed, turn never ended" leaves at unequal phases made passive lines
# look strictly better and the agent stopped attacking.
ROLLOUT_TURN_HORIZON: int = 2
MAX_ROLLOUT_DECISIONS: int = 30
MAX_ACTION_CANDIDATES: int = 10

# Kaggle gives each agent a per-game overage bank ("remainingOverageTime", ~600s
# with actTimeout=0); the agent is disqualified if it hits 0. The adaptive budget
# spreads the remaining bank over the expected remaining decisions.
MIN_DECISION_BUDGET_SECONDS: float = 0.3
MAX_DECISION_BUDGET_SECONDS: float = 2.5
LOW_OVERAGE_CUTOFF_SECONDS: float = 40.0
EXPECTED_DECISIONS_PER_GAME: int = 200
MIN_EXPECTED_REMAINING_DECISIONS: int = 40
# Retries when a sampled determinization fails the legality gate before falling back.
DETERMINIZATION_LEGALITY_RETRIES: int = 4

EVAL_WEIGHTS = {
    # Prize differential: dominant term (your prizes taken - opponent's prizes taken)
    "prize_diff": 100.0,
    "own_prizes_remaining": 0.0,
    # Active lethal-threat term (can KO active / active is lethal)
    "can_ko_active": 40.0,
    "active_in_danger": -35.0,
    # Board HP swing (sum of remaining-HP fractions, ours minus theirs; range ~[-6, 6])
    "hp_swing": 8.0,
    # Energy tempo: attached energy count + attack-affordability
    "energy_attached": 10.0,
    "attack_ready": 25.0,
    # Board development: bench fill, evolution status
    "bench_count": 8.0,
    "evolution_advantage": 15.0,
    # An empty bench after setup is one KO from losing outright — insurance
    # the turn-horizon rollout can't always see far enough to price.
    "benchless_penalty": -40.0,
    # Hand quality: hand size, draw/search supporters available
    "hand_size": 2.0,
    "supporter_in_hand": 8.0,
    # Deck-specific engine: Infernal Slash (Ceruledge #797) does NOTHING unless
    # 4 Basic Fire Energy are discarded from hand, so held Fire IS our damage.
    "fire_in_hand": 20.0,
    "infernal_ready": 60.0,
    # Status condition penalties (own active)
    "status_poisoned": -5.0,
    "status_burned": -5.0,
    "status_asleep": -15.0,
    "status_paralyzed": -25.0,
    "status_confused": -10.0,
}
