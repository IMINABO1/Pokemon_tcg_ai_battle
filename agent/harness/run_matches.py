"""Batch match runner with multiprocessing support.

Usage:
    python agent/harness/run_matches.py --n 16 --opponent greedy --workers 8
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
src_dir = root_dir / "agent" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def _ensure_paths():
    for p in (root_dir / "agent", src_dir, root_dir / "sample_submission" / "sample_submission"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _worker_run_match(match_args: tuple) -> dict:
    """Worker task run in a separate process (one match per process: the engine
    Battle is a per-process singleton and agent trackers are process-global)."""
    _ensure_paths()

    from harness.local_match import (
        run_one_match, random_agent, greedy_agent, agent_deck, sample_deck,
    )
    from ptcg_agent.policy import agent_decide

    match_id, type0, name0, type1, name1, overage_bank = match_args

    def resolve(agent_type):
        if agent_type == "pimc":
            return agent_decide, agent_deck()
        if agent_type == "greedy":
            return greedy_agent, agent_deck()
        return random_agent, sample_deck()

    func0, deck0 = resolve(type0)
    func1, deck1 = resolve(type1)

    res = run_one_match(deck0, func0, name0, deck1, func1, name1,
                        overage_bank=overage_bank)

    return {
        "match_id": match_id,
        "name0": name0,
        "name1": name1,
        "winner": res.winner,
        "winner_name": res.winner_name,
        "actions": res.total_actions,
        "duration": round(res.duration_seconds, 3),
        "p0_decisions": res.stats0.decisions,
        "p0_max_s": round(res.stats0.max_decision_seconds, 4),
        "p0_mean_s": round(res.stats0.mean_decision_seconds, 4),
        "p0_total_s": round(res.stats0.total_seconds, 2),
        "p1_decisions": res.stats1.decisions,
        "p1_max_s": round(res.stats1.max_decision_seconds, 4),
        "p1_mean_s": round(res.stats1.mean_decision_seconds, 4),
        "p1_total_s": round(res.stats1.total_seconds, 2),
        "error": res.error_message or "",
    }


def run_suite(
    num_matches: int = 16,
    max_workers: int = 8,
    opponent: str = "random",
    output_csv: str = "agent/harness/results/suite_results.csv",
    overage_bank: float = 600.0,
):
    """Run a suite of PIMC-vs-opponent matches, alternating seats."""
    opp_name = f"{opponent.capitalize()}_Agent"
    tasks = []
    for i in range(num_matches):
        if i % 2 == 0:
            tasks.append((i, "pimc", "PIMC_Agent", opponent, opp_name, overage_bank))
        else:
            tasks.append((i, opponent, opp_name, "pimc", "PIMC_Agent", overage_bank))

    print(f"Starting {num_matches} matches vs {opp_name} across {max_workers} workers...")
    results = []
    wins = {"PIMC_Agent": 0, opp_name: 0, "Draw": 0, "Unknown": 0}

    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
        futures = [executor.submit(_worker_run_match, task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            wins[res["winner_name"]] = wins.get(res["winner_name"], 0) + 1
            pimc_side = "p0" if res["name0"] == "PIMC_Agent" else "p1"
            print(
                f"Match {res['match_id']:3d}: {res['winner_name']:12s} in "
                f"{res['actions']:4d} actions ({res['duration']:7.1f}s) "
                f"pimc max/mean {res[f'{pimc_side}_max_s']:.2f}/{res[f'{pimc_side}_mean_s']:.2f}s"
                + (f"  ERROR: {res['error']}" if res["error"] else "")
            )

    n = len(results)
    pimc_decision_stats = []
    for r in results:
        side = "p0" if r["name0"] == "PIMC_Agent" else "p1"
        pimc_decision_stats.append((r[f"{side}_max_s"], r[f"{side}_mean_s"], r[f"{side}_total_s"]))
    overall_max = max((s[0] for s in pimc_decision_stats), default=0.0)
    overall_mean = (sum(s[1] for s in pimc_decision_stats) / n) if n else 0.0
    worst_total = max((s[2] for s in pimc_decision_stats), default=0.0)

    print("\n--- Suite Results ---")
    print(f"Total Matches: {n}")
    for name, w in wins.items():
        if w:
            print(f"{name}: {w} ({w / n * 100:.1f}%)")
    print(f"PIMC decision seconds: max={overall_max:.2f} mean-of-means={overall_mean:.2f} "
          f"worst game total={worst_total:.1f}s (bank {overage_bank:.0f}s)")
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"Matches with errors: {len(errors)}")

    results.sort(key=lambda r: r["match_id"])
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_csv}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--opponent", choices=["random", "greedy"], default="random")
    ap.add_argument("--csv", default="agent/harness/results/suite_results.csv")
    ap.add_argument("--overage", type=float, default=600.0)
    a = ap.parse_args()
    run_suite(a.n, a.workers, a.opponent, a.csv, a.overage)
