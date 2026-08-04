"""Batch match runner with multiprocessing support."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

# Ensure src is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
src_dir = root_dir / "agent" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ptcg_agent.policy import agent_decide, read_deck_csv
from ptcg_agent.carddb import validate_deck_legality


def _worker_run_match(match_args: tuple) -> dict:
    """Worker task run in a separate process."""
    agent_dir = root_dir / "agent"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(root_dir / "sample_submission" / "sample_submission") not in sys.path:
        sys.path.insert(0, str(root_dir / "sample_submission" / "sample_submission"))

    from harness.local_match import run_one_match, random_agent
    from ptcg_agent.policy import agent_decide
    from ptcg_agent import stats

    match_id, deck0, name0, agent_type0, deck1, name1, agent_type1 = match_args

    func0 = agent_decide if agent_type0 == "pimc" else random_agent
    func1 = agent_decide if agent_type1 == "pimc" else random_agent

    stats.reset()
    res = run_one_match(deck0, func0, name0, deck1, func1, name1)
    counters = stats.snapshot()

    decisions = counters.get("decisions", 0)
    fallback = counters.get("decision_fallback", 0)
    cand_total = counters.get("candidates_total", 0)
    cand_scored = counters.get("candidates_scored", 0)

    return {
        "match_id": match_id,
        "name0": name0,
        "name1": name1,
        "winner": res.winner,
        "winner_name": res.winner_name,
        "actions": res.total_actions,
        "duration": res.duration_seconds,
        "error": res.error_message or "",
        "decisions": decisions,
        "decision_fallback": fallback,
        "fallback_rate": round(fallback / decisions, 3) if decisions else 0.0,
        "search_decisions": counters.get("search_decisions", 0),
        "search_begin_ok": counters.get("search_begin_ok", 0),
        "search_begin_fail": counters.get("search_begin_fail", 0),
        "candidates_total": cand_total,
        "candidates_scored": cand_scored,
        "cand_scored_rate": round(cand_scored / cand_total, 3) if cand_total else 0.0,
        "rollout_fail": counters.get("rollout_fail", 0),
        "search_degenerate": counters.get("search_degenerate", 0),
        "search_picked_nonzero": counters.get("search_picked_nonzero", 0),
    }


def run_suite(
    num_matches: int = 10,
    max_workers: int = 4,
    agent0_type: str = "pimc",
    agent1_type: str = "random",
    output_csv: str = "agent/harness/results/suite_results.csv"
):
    """Run a suite of matches concurrently across worker processes."""
    pkg_deck = read_deck_csv()

    # Sample deck for random agent
    sample_deck_file = root_dir / "sample_submission" / "sample_submission" / "deck.csv"
    with open(sample_deck_file, "r") as f:
        sample_deck = [int(line.strip()) for line in f.read().split("\n") if line.strip()]

    tasks = []
    for i in range(num_matches):
        tasks.append(
            (i, pkg_deck, "PIMC_Agent", agent0_type, sample_deck, "Random_Agent", agent1_type)
        )

    print(f"Starting {num_matches} matches across {max_workers} worker processes...")
    results = []
    wins0 = 0
    wins1 = 0
    draws = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_run_match, task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if res["winner"] == 0:
                wins0 += 1
            elif res["winner"] == 1:
                wins1 += 1
            else:
                draws += 1
            print(f"Match {res['match_id']}: Winner = {res['winner_name']} in {res['actions']} actions ({res['duration']:.2f}s)")

    print(f"\n--- Suite Results ---")
    print(f"Total Matches: {len(results)}")
    print(f"PIMC_Agent Wins: {wins0} ({wins0/len(results)*100:.1f}%)")
    print(f"Random_Agent Wins: {wins1} ({wins1/len(results)*100:.1f}%)")
    print(f"Draws: {draws}")

    tot_dec = sum(r["decisions"] for r in results)
    tot_fb = sum(r["decision_fallback"] for r in results)
    tot_ct = sum(r["candidates_total"] for r in results)
    tot_cs = sum(r["candidates_scored"] for r in results)
    print(f"\n--- Observability (Phase 0) ---")
    print(f"Decisions: {tot_dec} | Fallbacks: {tot_fb} ({tot_fb/tot_dec*100:.1f}% of decisions)" if tot_dec else "Decisions: 0")
    print(f"search_begin ok/fail: {sum(r['search_begin_ok'] for r in results)}/{sum(r['search_begin_fail'] for r in results)}")
    print(f"Candidates scored/total: {tot_cs}/{tot_ct} ({tot_cs/tot_ct*100:.1f}% scored)" if tot_ct else "Candidates: 0 (search never reached rollout)")
    print(f"rollout_fail: {sum(r['rollout_fail'] for r in results)}")
    tot_sd = sum(r["search_decisions"] for r in results)
    tot_deg = sum(r["search_degenerate"] for r in results)
    tot_nz = sum(r["search_picked_nonzero"] for r in results)
    if tot_sd:
        print(f"Search decisions: {tot_sd} | degenerate ties: {tot_deg} ({tot_deg/tot_sd*100:.1f}%) | picked non-default: {tot_nz} ({tot_nz/tot_sd*100:.1f}%)")

    # Save to CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "match_id", "name0", "name1", "winner", "winner_name", "actions", "duration", "error",
            "decisions", "decision_fallback", "fallback_rate", "search_decisions",
            "search_begin_ok", "search_begin_fail", "candidates_total", "candidates_scored",
            "cand_scored_rate", "rollout_fail", "search_degenerate", "search_picked_nonzero",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    run_suite(num_matches=5, max_workers=2)
