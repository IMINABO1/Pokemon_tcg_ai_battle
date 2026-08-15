"""Replay JSON -> per-player decision records (JSONL, one file per game).

THE LEAKAGE RULE (enforced here and re-audited in audit_leakage.py):
For a record belonging to player P at step t, model INPUTS may only contain
what P's own agent could see at t: P's observation, P's prior observations and
actions, and public history. The opponent's private info and anything revealed
after t is FORBIDDEN as input; the opponent's end-of-game revealed deck and
archetype are stored under `aux_*` fields as auxiliary TARGETS only.

Every `# ADAPT:` marker is a spot where the real replay schema (from
inspect_replay.py) must be confirmed before trusting bulk output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- helpers

def deck_fingerprint(card_ids: list[int]) -> str:
    return hashlib.sha1(",".join(map(str, sorted(card_ids))).encode()).hexdigest()[:16]


class BeliefTracker:
    """Engineered belief features computable strictly from one player's own
    observation stream. Mirrors what determinize.py does at inference, so
    every feature here is reproducible live (the leakage litmus test)."""

    def __init__(self):
        self.opp_revealed = Counter()      # card_id -> times seen (played/discarded/attached)
        self.opp_energy_attachments = 0
        self.opp_attach_opportunities = 0
        self.opp_supporters_seen = Counter()
        self.opp_evolutions_seen = 0

    def update_from_public_log(self, log_events: list[dict]) -> None:
        # ADAPT: event type strings/fields must match real obs['logs'] entries.
        for ev in log_events:
            actor = ev.get("player")
            if actor != "opponent":
                continue
            et = ev.get("type", "")
            cid = ev.get("card_id")
            if cid is not None:
                self.opp_revealed[cid] += 1
            if et == "ENERGY_ATTACH":
                self.opp_energy_attachments += 1
            elif et == "SUPPORTER_PLAYED":
                self.opp_supporters_seen[cid] += 1
            elif et == "EVOLVE":
                self.opp_evolutions_seen += 1

    def features(self) -> dict:
        return {
            "revealed_opponent_cards": dict(self.opp_revealed),
            "opp_energy_attachments": self.opp_energy_attachments,
            "opp_attach_opportunities": self.opp_attach_opportunities,
            "supporters_seen": dict(self.opp_supporters_seen),
            "evolutions_seen": self.opp_evolutions_seen,
        }


# ---------------------------------------------------------------- core

def parse_one(replay: dict, game_id: str) -> tuple[dict, list[dict]]:
    """Returns (game_row, decision_rows)."""
    # ADAPT: confirm top-level shape. kaggle-environments replays:
    #   replay['steps'][t][agent_idx] = {observation, action, reward, status, info}
    #   replay['info'] may carry TeamNames / submission ids / final ratings.
    steps = replay["steps"]
    info = replay.get("info", {})

    # ADAPT: metadata field names below against real replay['info'].
    meta = {
        "game_id": game_id,
        "timestamp": replay.get("timestamp") or info.get("EndTime"),
        "player_0_submission": info.get("SubmissionIds", [None, None])[0],
        "player_1_submission": info.get("SubmissionIds", [None, None])[1],
        "player_0_team": info.get("TeamNames", [None, None])[0],
        "player_1_team": info.get("TeamNames", [None, None])[1],
        # Ratings at episode time — used for sample weighting, NEVER as input.
        "player_0_mu": info.get("Mu", [None, None])[0] if "Mu" in info else None,
        "player_1_mu": info.get("Mu", [None, None])[1] if "Mu" in info else None,
        "player_0_sigma": info.get("Sigma", [None, None])[0] if "Sigma" in info else None,
        "player_1_sigma": info.get("Sigma", [None, None])[1] if "Sigma" in info else None,
    }

    # Outcome from final rewards. ADAPT: confirm reward convention.
    final = steps[-1]
    r0 = final[0].get("reward")
    r1 = final[1].get("reward")
    meta["winner"] = 0 if (r0 or 0) > (r1 or 0) else 1 if (r1 or 0) > (r0 or 0) else -1
    meta["termination_reason"] = final[0].get("status")

    decisions: list[dict] = []
    trackers = [BeliefTracker(), BeliefTracker()]
    decks: list[list[int] | None] = [None, None]

    for t, agents in enumerate(steps):
        for p in (0, 1):
            rec = agents[p]
            obs = rec.get("observation") or {}
            action = rec.get("action")
            if action is None:
                continue
            select = obs.get("select")
            # Deck-selection step: select and current both None, action = 60 ids.
            if select is None and obs.get("current") is None and isinstance(action, list) and len(action) == 60:
                decks[p] = list(action)
                continue
            if select is None:
                continue

            trackers[p].update_from_public_log(obs.get("logs", []) or [])
            legal = select.get("option", [])
            decisions.append({
                "game_id": game_id,
                "player": p,
                "decision_index": len([d for d in decisions if d["player"] == p]),
                "step": t,
                "turn": obs.get("turn"),
                "select_type": select.get("type"),
                # -------- STRICT LEGAL INPUTS --------
                "observation": obs,             # the player's own full obs at t
                "belief_features": trackers[p].features(),
                "own_deck_ids": decks[p],
                "legal_options": legal,
                # -------- SUPERVISION --------
                "chosen_indices": action if isinstance(action, list) else [action],
                "total_step": t,  # remaining_turns derived after game length known
                # outcome/mu/opponent_mu filled by build_dataset (targets/weights)
            })

    meta["num_decisions"] = len(decisions)
    meta["player_0_deck_fingerprint"] = deck_fingerprint(decks[0]) if decks[0] else None
    meta["player_1_deck_fingerprint"] = deck_fingerprint(decks[1]) if decks[1] else None
    meta["player_0_deck_ids"] = decks[0]
    meta["player_1_deck_ids"] = decks[1]
    return meta, decisions


def main(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    games_fp = out_dir / "games.jsonl"
    n_ok = n_bad = 0
    with games_fp.open("a") as gf:
        for fp in sorted(raw_dir.rglob("*.json")):
            game_id = fp.stem
            dest = out_dir / f"{game_id}.decisions.jsonl"
            if dest.exists():
                continue
            try:
                replay = json.loads(fp.read_text())
                meta, decisions = parse_one(replay, game_id)
            except Exception as e:  # noqa: BLE001
                n_bad += 1
                print(f"FAIL {fp}: {e}")
                continue
            with dest.open("w") as df:
                for d in decisions:
                    df.write(json.dumps(d) + "\n")
            gf.write(json.dumps(meta) + "\n")
            n_ok += 1
    print(f"parsed ok={n_ok} failed={n_bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data_lake/raw")
    ap.add_argument("--out", default="data_lake/parsed")
    a = ap.parse_args()
    main(Path(a.raw), Path(a.out))
