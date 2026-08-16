"""parsed JSONL -> games.parquet + decisions.parquet.

Adds: outcome labels, sample weights (weighting.py), deck-archetype clustering
(fingerprint-based, refine later), aux_* targets, dedup, and a time-based
train/val split (val = most recent K days: the model must generalize FORWARD
in a shifting meta, so a random split would lie to you).
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from weighting import combine

VAL_LAST_DAYS = 3


def load_jsonl(fp: Path):
    with fp.open() as f:
        for line in f:
            yield json.loads(line)


# Card ids 1-8 are the basic energies; every deck maxes them, so they say
# nothing about archetype.
BASIC_ENERGY_IDS = set(range(1, 9))


def archetype_of(deck_ids: list[int] | None, card_names: dict[int, str] | None) -> str:
    """v0: archetype = most-copied non-basic-energy card name; falls back to
    fingerprint prefix. Replace with proper clustering once carddb is wired."""
    if not deck_ids:
        return "unknown"
    counts = Counter(cid for cid in deck_ids if cid not in BASIC_ENERGY_IDS)
    if not counts:
        return "unknown"
    top_id, _ = counts.most_common(1)[0]
    if card_names and top_id in card_names:
        return card_names[top_id]
    return f"deck_{top_id}"


def main(parsed: Path, out: Path, carddb_csv: Path | None) -> None:
    card_names = None
    if carddb_csv and carddb_csv.exists():
        df = pd.read_csv(carddb_csv)
        # ADAPT: confirm column names in EN_Card_Data.csv
        id_col = next(c for c in df.columns if "id" in c.lower())
        nm_col = next(c for c in df.columns if "name" in c.lower())
        card_names = dict(zip(df[id_col], df[nm_col]))

    games = pd.DataFrame(list(load_jsonl(parsed / "games.jsonl")))
    games = games.drop_duplicates(subset="game_id")
    games["ts"] = pd.to_datetime(games["timestamp"], errors="coerce", utc=True)
    now = games["ts"].max() if games["ts"].notna().any() else pd.Timestamp.now(tz="UTC")

    for p in (0, 1):
        games[f"player_{p}_archetype"] = games[f"player_{p}_deck_ids"].apply(
            lambda d: archetype_of(d, card_names)
        )

    # archetype frequency for diversity weighting
    arch_counts = Counter(games["player_0_archetype"]) + Counter(games["player_1_archetype"])
    total = sum(arch_counts.values())
    n_arch = len(arch_counts)

    rows = []
    for _, g in games.iterrows():
        age = (now - g["ts"]).days if pd.notna(g["ts"]) else 30
        for p in (0, 1):
            rows.append({
                "game_id": g["game_id"],
                "player": p,
                "weight": combine(
                    g.get(f"player_{p}_mu"), g.get(f"player_{p}_sigma"),
                    age, arch_counts[g[f"player_{p}_archetype"]], total, n_arch,
                ),
                "outcome": 1.0 if g["winner"] == p else (-1.0 if g["winner"] in (0, 1) else 0.0),
                "aux_opponent_archetype": g[f"player_{1 - p}_archetype"],
                "aux_opponent_deck_ids": g[f"player_{1 - p}_deck_ids"],
                "mu": g.get(f"player_{p}_mu"),
                "opponent_mu": g.get(f"player_{1 - p}_mu"),
                "num_steps": g.get("num_decisions"),
            })
    pw = pd.DataFrame(rows)

    # decisions
    dec_rows = []
    for fp in sorted(parsed.glob("*.decisions.jsonl")):
        dec_rows.extend(load_jsonl(fp))
    dec = pd.DataFrame(dec_rows)
    dec = dec.merge(pw, on=["game_id", "player"], how="left")
    if "total_step" in dec.columns:
        dec["remaining_turns"] = (dec["num_steps"] - dec["total_step"]).clip(lower=0)

    # serialize nested cols for parquet
    for col in ("observation", "belief_features", "legal_options",
                "chosen_indices", "own_deck_ids", "aux_opponent_deck_ids"):
        if col in dec.columns:
            dec[col] = dec[col].apply(json.dumps)

    # forward-in-time split
    games["split"] = "train"
    cutoff = now - dt.timedelta(days=VAL_LAST_DAYS)
    games.loc[games["ts"] >= cutoff, "split"] = "val"
    dec = dec.merge(games[["game_id", "split"]], on="game_id", how="left")

    out.mkdir(parents=True, exist_ok=True)
    games.drop(columns=["player_0_deck_ids", "player_1_deck_ids"]).to_parquet(out / "games.parquet")
    dec.to_parquet(out / "decisions.parquet")
    print(f"games={len(games)} decisions={len(dec)} "
          f"val_frac={(dec['split'] == 'val').mean():.3f}")
    print("archetype distribution:", dict(arch_counts.most_common(12)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data_lake/parsed")
    ap.add_argument("--out", default="data_lake")
    ap.add_argument("--carddb", default="EN_Card_Data.csv")
    a = ap.parse_args()
    main(Path(a.parsed), Path(a.out), Path(a.carddb))
