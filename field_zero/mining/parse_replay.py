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


# Engine LogType values (cg/api.py): a log entry is
# {type: int, playerIndex: int, cardId: int, serial: int, ...}
LOG_PLAY = 10
LOG_ATTACH = 11
LOG_EVOLVE = 12


class BeliefTracker:
    """Engineered belief features computable strictly from one player's own
    observation stream. Mirrors what determinize.py does at inference, so
    every feature here is reproducible live (the leakage litmus test)."""

    def __init__(self):
        # serial -> card_id: the same physical card is re-logged on every move,
        # so revealed-card counts must dedup by serial (matches determinize.py).
        self.opp_seen_serials: dict[int, int] = {}
        self.opp_energy_attachments = 0
        self.opp_plays = 0
        self.opp_evolutions_seen = 0

    def update_from_public_log(self, log_events: list[dict], opp_idx: int) -> None:
        for ev in log_events:
            if ev.get("playerIndex") != opp_idx:
                continue
            et = ev.get("type")
            cid = ev.get("cardId")
            serial = ev.get("serial")
            if cid and serial is not None:
                self.opp_seen_serials[serial] = cid
            if et == LOG_ATTACH:
                self.opp_energy_attachments += 1
            elif et == LOG_PLAY:
                self.opp_plays += 1
            elif et == LOG_EVOLVE:
                self.opp_evolutions_seen += 1

    def features(self) -> dict:
        revealed = Counter(self.opp_seen_serials.values())
        return {
            "revealed_opponent_cards": dict(revealed),
            "opp_energy_attachments": self.opp_energy_attachments,
            "opp_plays": self.opp_plays,
            "evolutions_seen": self.opp_evolutions_seen,
        }


# ---------------------------------------------------------------- core

def parse_one(replay: dict, game_id: str, game_date: str | None = None) -> tuple[dict, list[dict]]:
    """Returns (game_row, decision_rows).

    Real replay['info'] carries only Agents/EpisodeId/TeamNames — no ratings or
    submission ids. Ratings for weighting must be joined later from Kaggle's
    episode-metadata tables; weighting.py already handles mu=None.
    """
    steps = replay["steps"]
    info = replay.get("info", {})
    teams = info.get("TeamNames", [None, None])

    meta = {
        "game_id": game_id,
        # Replays carry no timestamp; the daily dataset date is the best proxy.
        "timestamp": game_date,
        "episode_id": info.get("EpisodeId"),
        "player_0_team": teams[0] if len(teams) > 0 else None,
        "player_1_team": teams[1] if len(teams) > 1 else None,
        "player_0_mu": None,
        "player_1_mu": None,
        "player_0_sigma": None,
        "player_1_sigma": None,
    }

    # Outcome: top-level rewards are [-1, 1] style (Lost:-1, Won:1, Draw:0).
    rewards = replay.get("rewards") or [None, None]
    r0, r1 = rewards[0], rewards[1]
    meta["winner"] = 0 if (r0 or 0) > (r1 or 0) else 1 if (r1 or 0) > (r0 or 0) else -1
    statuses = replay.get("statuses") or []
    meta["termination_reason"] = statuses[0] if statuses else None

    decisions: list[dict] = []
    trackers = [BeliefTracker(), BeliefTracker()]
    decks: list[list[int] | None] = [None, None]

    per_player_count = [0, 0]
    # kaggle-environments pairing: the action recorded at steps[t] was produced
    # from the observation recorded at steps[t-1], and only by the agent whose
    # status at t-1 is ACTIVE (INACTIVE agents echo a stale obs and []).
    for t in range(1, len(steps)):
        for p in (0, 1):
            if steps[t - 1][p].get("status") != "ACTIVE":
                continue
            obs = steps[t - 1][p].get("observation") or {}
            action = steps[t][p].get("action")
            if action is None:
                continue
            select = obs.get("select")
            # Deck-selection step: select and current both None, action = 60 ids.
            if select is None and obs.get("current") is None and isinstance(action, list) and len(action) == 60:
                decks[p] = list(action)
                continue
            if select is None:
                continue

            trackers[p].update_from_public_log(obs.get("logs", []) or [], opp_idx=1 - p)
            current = obs.get("current") or {}
            legal = select.get("option", [])
            decisions.append({
                "game_id": game_id,
                "player": p,
                "decision_index": per_player_count[p],
                "step": t,
                "turn": current.get("turn"),
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
            per_player_count[p] += 1

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
                # Raw layout is data_lake/raw/<YYYY-MM-DD>/<episode>.json; the
                # folder date stands in for the missing replay timestamp.
                date = fp.parent.name if fp.parent.name[:2] == "20" else None
                meta, decisions = parse_one(replay, game_id, game_date=date)
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
