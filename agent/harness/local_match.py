"""Local match runner using real C++ engine bindings."""
from dataclasses import dataclass
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
from cg.api import to_observation_class


@dataclass
class MatchResult:
    winner: int  # 0, 1, or 2 (draw)
    total_actions: int
    duration_seconds: float
    error_message: Optional[str] = None
    winner_name: str = ""


def random_agent(obs_dict: dict) -> list[int]:
    """Sample random agent for baseline benchmarking."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial deck selection
        file_path = root_dir / "sample_submission" / "sample_submission" / "deck.csv"
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.read().split("\n") if line.strip()]
        return [int(lines[i]) for i in range(60)]

    n_options = len(obs.select.option)
    min_c = obs.select.minCount
    max_c = obs.select.maxCount

    count = random.randint(min_c, max_c)
    if count == 0:
        return []
    return random.sample(list(range(n_options)), count)


def run_one_match(
    deck0: list[int],
    agent0: Callable[[dict], list[int]],
    name0: str,
    deck1: list[int],
    agent1: Callable[[dict], list[int]],
    name1: str,
    max_actions: int = 3000
) -> MatchResult:
    """Run a single local match between two agents."""
    start_time = time.monotonic()
    actions = 0
    err_msg = None
    winner = -1

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

            try:
                choice = active_agent(obs)
            except Exception as e:
                winner = 1 - current_p
                err_msg = f"Exception in player {current_p}: {e}"
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
        winner_name=wname
    )
