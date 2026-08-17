"""parsed JSONL -> games.parquet + decisions.parquet, STREAMING.

Adds: outcome labels, sample weights (weighting.py), deck-archetype clustering
(fingerprint-based, refine later), aux_* targets, dedup, and a time-based
train/val split (val = most recent K days: the model must generalize FORWARD
in a shifting meta, so a random split would lie to you).

Memory model: the games table (one row per game) fits in RAM at any realistic
scale; the decisions table does NOT (25M+ rows with embedded observations at
~70k games ate >260 GB and OOM-killed three cluster jobs). Decisions are
therefore streamed game-file-by-game-file and written to parquet in bounded
chunks — peak RAM is O(chunk), independent of dataset size.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from weighting import combine

VAL_LAST_DAYS = 3
CHUNK_ROWS = 100_000

# Card ids 1-8 are the basic energies; every deck maxes them, so they say
# nothing about archetype.
BASIC_ENERGY_IDS = set(range(1, 9))

DECISION_COLUMNS = [
    "game_id", "player", "decision_index", "step", "turn", "select_type",
    "observation", "belief_features", "own_deck_ids", "legal_options",
    "chosen_indices", "total_step", "weight", "outcome",
    "aux_opponent_archetype", "aux_opponent_deck_ids", "mu", "opponent_mu",
    "num_steps", "remaining_turns", "split",
]

JSON_COLUMNS = ("observation", "belief_features", "legal_options",
                "chosen_indices", "own_deck_ids", "aux_opponent_deck_ids")


def load_jsonl(fp: Path):
    with fp.open() as f:
        for line in f:
            yield json.loads(line)


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


def build_games(parsed: Path, card_names: dict[int, str] | None) -> pd.DataFrame:
    games = pd.DataFrame(list(load_jsonl(parsed / "games.jsonl")))
    games = games.drop_duplicates(subset="game_id")
    games["ts"] = pd.to_datetime(games["timestamp"], errors="coerce", utc=True)
    now = games["ts"].max() if games["ts"].notna().any() else pd.Timestamp.now(tz="UTC")

    for p in (0, 1):
        games[f"player_{p}_archetype"] = games[f"player_{p}_deck_ids"].apply(
            lambda d: archetype_of(d, card_names)
        )

    games["split"] = "train"
    cutoff = now - dt.timedelta(days=VAL_LAST_DAYS)
    games.loc[games["ts"] >= cutoff, "split"] = "val"
    return games, now


def per_player_rows(games: pd.DataFrame, now) -> dict[tuple[str, int], dict]:
    """Small lookup: (game_id, player) -> weight/outcome/aux fields."""
    arch_counts = Counter(games["player_0_archetype"]) + Counter(games["player_1_archetype"])
    total = sum(arch_counts.values())
    n_arch = len(arch_counts)

    lookup = {}
    for _, g in games.iterrows():
        age = (now - g["ts"]).days if pd.notna(g["ts"]) else 30
        for p in (0, 1):
            lookup[(g["game_id"], p)] = {
                "weight": combine(
                    g.get(f"player_{p}_mu"), g.get(f"player_{p}_sigma"),
                    age, arch_counts[g[f"player_{p}_archetype"]], total, n_arch,
                ),
                "outcome": 1.0 if g["winner"] == p else (-1.0 if g["winner"] in (0, 1) else 0.0),
                "aux_opponent_archetype": g[f"player_{1 - p}_archetype"],
                "aux_opponent_deck_ids": g.get(f"player_{1 - p}_deck_ids"),
                "mu": g.get(f"player_{p}_mu"),
                "opponent_mu": g.get(f"player_{1 - p}_mu"),
                "num_steps": g.get("num_decisions"),
                "split": g["split"],
            }
    print(f"archetype distribution: {dict(arch_counts.most_common(12))}")
    return lookup


def stream_decisions(parsed: Path, out: Path, lookup: dict) -> tuple[int, int]:
    writer = None
    buffer: list[dict] = []
    n_rows = n_val = 0

    def flush():
        nonlocal writer, buffer
        if not buffer:
            return
        df = pd.DataFrame(buffer, columns=DECISION_COLUMNS)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out / "decisions.parquet", table.schema)
        writer.write_table(table)
        buffer = []

    for fp in sorted(parsed.glob("*.decisions.jsonl")):
        for d in load_jsonl(fp):
            key = (d["game_id"], d["player"])
            extra = lookup.get(key)
            if extra is None:
                continue
            d.update(extra)
            ns = d.get("num_steps")
            ts = d.get("total_step")
            d["remaining_turns"] = max(0, ns - ts) if ns is not None and ts is not None else None
            for col in JSON_COLUMNS:
                if col in d and d[col] is not None and not isinstance(d[col], str):
                    d[col] = json.dumps(d[col])
            buffer.append({c: d.get(c) for c in DECISION_COLUMNS})
            n_rows += 1
            n_val += d["split"] == "val"
            if len(buffer) >= CHUNK_ROWS:
                flush()
    flush()
    if writer is not None:
        writer.close()
    return n_rows, n_val


def main(parsed: Path, out: Path, carddb_csv: Path | None) -> None:
    card_names = None
    if carddb_csv and carddb_csv.exists():
        df = pd.read_csv(carddb_csv)
        id_col = next(c for c in df.columns if "id" in c.lower())
        nm_col = next(c for c in df.columns if "name" in c.lower())
        card_names = dict(zip(df[id_col], df[nm_col]))

    out.mkdir(parents=True, exist_ok=True)
    games, now = build_games(parsed, card_names)
    lookup = per_player_rows(games, now)
    games.drop(columns=["player_0_deck_ids", "player_1_deck_ids"], errors="ignore") \
        .to_parquet(out / "games.parquet")

    n_rows, n_val = stream_decisions(parsed, out, lookup)
    print(f"games={len(games)} decisions={n_rows} "
          f"val_frac={(n_val / n_rows) if n_rows else 0:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data_lake/parsed")
    ap.add_argument("--out", default="data_lake")
    ap.add_argument("--carddb", default="EN_Card_Data.csv")
    a = ap.parse_args()
    main(Path(a.parsed), Path(a.out), Path(a.carddb))
