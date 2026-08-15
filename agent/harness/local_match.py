"""Local match runner using real C++ engine bindings."""
from dataclasses import dataclass, field
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Ensure cg is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
possible_cg = root_dir / "sample_submission" / "sample_submission"
if possible_cg.exists() and str(possible_cg) not in sys.path:
    sys.path.insert(0, str(possible_cg))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType

# Mirrors the ladder's per-agent banked thinking time (actTimeout=0, bank ~600s).
DEFAULT_OVERAGE_BANK_SECONDS = 600.0


@dataclass
class AgentStats:
    decisions: int = 0
    total_seconds: float = 0.0
    max_decision_seconds: float = 0.0
    durations: list = field(default_factory=list)

    @property
    def mean_decision_seconds(self) -> float:
        return self.total_seconds / self.decisions if self.decisions else 0.0

    @property
    def p95_decision_seconds(self) -> float:
        if not self.durations:
            return 0.0
        return sorted(self.durations)[int(0.95 * (len(self.durations) - 1))]


@dataclass
class MatchResult:
    winner: int  # 0, 1, or 2 (draw)
    total_actions: int
    duration_seconds: float
    error_message: Optional[str] = None
    winner_name: str = ""
    stats0: AgentStats = field(default_factory=AgentStats)
    stats1: AgentStats = field(default_factory=AgentStats)


def _read_deck(path: Path) -> list[int]:
    with open(path, "r") as f:
        lines = [line.strip() for line in f.read().split("\n") if line.strip()]
    return [int(lines[i]) for i in range(60)]


def sample_deck() -> list[int]:
    return _read_deck(root_dir / "sample_submission" / "sample_submission" / "deck.csv")


def agent_deck() -> list[int]:
    import os
    override = os.environ.get("PTCG_DECK")
    if override:
        return _read_deck(Path(override))
    return _read_deck(root_dir / "agent" / "src" / "ptcg_agent" / "deck.csv")


def random_agent(obs_dict: dict) -> list[int]:
    """Uniform-random baseline (floor check)."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return sample_deck()

    n_options = len(obs.select.option)
    min_c = obs.select.minCount
    max_c = min(obs.select.maxCount, n_options)

    count = random.randint(min_c, max_c)
    if count == 0:
        return []
    return random.sample(list(range(n_options)), count)


_GREEDY_MAIN_PRIORITY = {
    OptionType.ATTACK: 6,
    OptionType.ATTACH: 5,
    OptionType.EVOLVE: 4,
    OptionType.PLAY: 3,
    OptionType.ABILITY: 2,
    OptionType.YES: 1,
    OptionType.END: 0,
    OptionType.RETREAT: -1,
}


def greedy_agent(obs_dict: dict) -> list[int]:
    """Search-free aggressive baseline: attack when possible, otherwise develop.
    The discriminating opponent for benchmarking (random is only a floor)."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return agent_deck()

    select = obs.select
    options = select.option
    n_options = len(options)
    min_c = select.minCount
    max_c = min(select.maxCount, n_options)

    if n_options == 0 or max_c == 0:
        return []

    if min_c <= 1 <= max_c:
        best_i, best_p = 0, None
        for i, opt in enumerate(options):
            p = _GREEDY_MAIN_PRIORITY.get(opt.type, 0)
            if opt.type == OptionType.NUMBER:
                p = opt.number if opt.number is not None else 0
            if best_p is None or p > best_p:
                best_i, best_p = i, p
        return [best_i]

    count = max(min_c, min(1, max_c)) if min_c > 0 else max_c
    count = min(count, max_c)
    return list(range(count))


def run_one_match(
    deck0: list[int],
    agent0: Callable[[dict], list[int]],
    name0: str,
    deck1: list[int],
    agent1: Callable[[dict], list[int]],
    name1: str,
    max_actions: int = 3000,
    overage_bank: Optional[float] = DEFAULT_OVERAGE_BANK_SECONDS,
) -> MatchResult:
    """Run a single local match between two agents.

    When overage_bank is set, each agent sees a ladder-style
    `remainingOverageTime` in its obs dict, decremented by its own decision
    wall time; hitting 0 loses the match (mirrors Kaggle's TIMEOUT DQ).
    """
    start_time = time.monotonic()
    actions = 0
    err_msg = None
    winner = -1
    stats = (AgentStats(), AgentStats())
    banks = [overage_bank, overage_bank]

    try:
        obs, start_data = battle_start(deck0, deck1)
        if not obs or not start_data or start_data.battlePtr == 0:
            return MatchResult(
                winner=-1,
                total_actions=0,
                duration_seconds=time.monotonic() - start_time,
                error_message="Failed to start battle (invalid deck or engine error)."
            )

        while obs and obs.get("current", {}).get("result", -1) == -1:
            if actions >= max_actions:
                err_msg = "Match hit max_actions limit."
                break

            current_p = obs["current"]["yourIndex"]
            active_agent = agent0 if current_p == 0 else agent1

            obs_for_agent = obs
            if banks[current_p] is not None:
                obs_for_agent = dict(obs)
                obs_for_agent["remainingOverageTime"] = banks[current_p]

            t0 = time.monotonic()
            try:
                choice = active_agent(obs_for_agent)
            except Exception as e:
                winner = 1 - current_p
                err_msg = f"Exception in player {current_p}: {e}"
                break
            dt = time.monotonic() - t0

            st = stats[current_p]
            st.decisions += 1
            st.total_seconds += dt
            st.durations.append(dt)
            if dt > st.max_decision_seconds:
                st.max_decision_seconds = dt
            if banks[current_p] is not None:
                banks[current_p] -= dt
                if banks[current_p] < 0:
                    winner = 1 - current_p
                    err_msg = f"Player {current_p} exhausted overage bank (TIMEOUT)."
                    break

            try:
                obs = battle_select(choice)
            except Exception as e:
                winner = 1 - current_p
                err_msg = f"Engine error on select for player {current_p}: {e}"
                break

            actions += 1

        if winner == -1 and obs and "current" in obs:
            winner = obs["current"].get("result", -1)

    except Exception as e:
        err_msg = f"Unhandled match exception: {e}"
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    duration = time.monotonic() - start_time
    wname = name0 if winner == 0 else (name1 if winner == 1 else ("Draw" if winner == 2 else "Unknown"))

    return MatchResult(
        winner=winner,
        total_actions=actions,
        duration_seconds=duration,
        error_message=err_msg,
        winner_name=wname,
        stats0=stats[0],
        stats1=stats[1],
    )
