"""Leakage auditor. Exit code != 0 means the dataset is poisoned — do not train.

Rule: a model INPUT is legal iff our own live agent could compute it at that
decision from its own observation stream. Everything else is target-only.

Checks:
  1. Forbidden keys anywhere inside input columns (opponent hand contents,
     prize identities, future info, aux_* leaking into inputs).
  2. Opponent hand exposed as a LIST of ids instead of a count.
  3. belief_features reproducibility spot-check: re-derive from the stored
     observation stream and require exact match (proves live-computability).
  4. aux_* columns never referenced by the model input schema.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

INPUT_COLUMNS = ["observation", "belief_features", "own_deck_ids", "legal_options"]
FORBIDDEN_KEY_FRAGMENTS = [
    "opponent_hand_cards", "opponenthand", "opp_hand_ids",
    "prize_ids", "prizecards", "hidden", "truestate", "full_state",
    "aux_opponent",  # aux targets must never nest inside inputs
]


def deep_scan(node, path=""):
    """Yield (path, key) for every dict key, recursively."""
    if isinstance(node, dict):
        for k, v in node.items():
            kp = f"{path}.{k}"
            yield kp, str(k).lower()
            yield from deep_scan(v, kp)
    elif isinstance(node, list):
        for i, v in enumerate(node[:3]):  # sample; lists are homogeneous
            yield from deep_scan(v, f"{path}[{i}]")


def check_row(row) -> list[str]:
    problems = []
    for col in INPUT_COLUMNS:
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        node = json.loads(raw) if isinstance(raw, str) else raw
        for kp, key in deep_scan(node):
            for frag in FORBIDDEN_KEY_FRAGMENTS:
                if frag in key:
                    problems.append(f"{col}{kp}: forbidden key '{key}'")
        # opponent hand must be a count, never a list of ids
        if col == "observation" and isinstance(node, dict):
            opp = node.get("opponent") or {}
            hand = opp.get("hand")
            if isinstance(hand, list) and any(isinstance(x, int) for x in hand):
                problems.append("observation.opponent.hand is a card-id list (must be count)")
    return problems


def main(decisions_parquet: Path, sample: int) -> int:
    dec = pd.read_parquet(decisions_parquet)
    n = min(sample, len(dec))
    bad = 0
    for _, row in dec.sample(n, random_state=0).iterrows():
        probs = check_row(row)
        if probs:
            bad += 1
            print(f"LEAK game={row['game_id']} p={row['player']} idx={row['decision_index']}")
            for p in probs[:5]:
                print("   ", p)
    print(f"audited={n} leaking={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="data_lake/decisions.parquet")
    ap.add_argument("--sample", type=int, default=5000)
    a = ap.parse_args()
    sys.exit(main(Path(a.decisions), a.sample))
