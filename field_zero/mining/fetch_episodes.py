"""Fetch PTCG AI Battle episode replays.

Two official sources, both supported:
  A) Daily episode datasets: kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD
     (discovered via kaggle/pokemon-tcg-ai-battle-episodes-index)
  B) Kaggle Episodes API (ListEpisodes / GetEpisodeReplay) for targeted pulls
     of a specific submission's games.

Usage:
  python fetch_episodes.py --since 2026-07-01 --out data_lake/raw
  python fetch_episodes.py --submission-id 53989933 --out data_lake/raw  # targeted
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

INDEX_DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DAILY_TEMPLATE = "kaggle/pokemon-tcg-ai-battle-episodes-{date}"  # YYYY-MM-DD


def sh(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def download_dataset(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    rc = sh(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"])
    return rc == 0


def daterange(since: dt.date, until: dt.date):
    d = since
    while d <= until:
        yield d
        d += dt.timedelta(days=1)


def fetch_daily(since: str, until: str | None, out: Path) -> None:
    # Index first: tells us which dates actually exist.
    idx_dir = out / "_index"
    download_dataset(INDEX_DATASET, idx_dir)

    start = dt.date.fromisoformat(since)
    end = dt.date.fromisoformat(until) if until else dt.date.today()
    ok, missing = [], []
    for d in daterange(start, end):
        slug = DAILY_TEMPLATE.format(date=d.isoformat())
        dest = out / d.isoformat()
        if dest.exists() and any(dest.iterdir()):
            print(f"skip {d} (already present)")
            continue
        if download_dataset(slug, dest):
            ok.append(str(d))
        else:
            missing.append(str(d))
        time.sleep(1.0)  # be polite; kaggle rate-limits
    print(f"\nfetched={len(ok)} missing={missing}")


def fetch_episode_replays_for_submission(submission_id: int, out: Path) -> None:
    """Targeted pull via the (unofficial-but-stable) Episodes API used by
    kaggle-environments meta. Lists episodes for one SubmissionId, then pulls
    each replay JSON."""
    import urllib.request

    api = "https://www.kaggle.com/api/i/competitions.EpisodeService/"

    def post(endpoint: str, body: dict) -> dict:
        req = urllib.request.Request(
            api + endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    listing = post("ListEpisodes", {"submissionId": submission_id})
    eps = listing.get("episodes", [])
    print(f"submission {submission_id}: {len(eps)} episodes")
    dest = out / f"submission_{submission_id}"
    dest.mkdir(parents=True, exist_ok=True)
    for ep in eps:
        eid = ep["id"]
        fp = dest / f"{eid}.json"
        if fp.exists():
            continue
        replay = post("GetEpisodeReplay", {"episodeId": eid})
        fp.write_text(json.dumps(replay))
        time.sleep(0.5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2026-06-01")
    p.add_argument("--until", default=None)
    p.add_argument("--out", default="data_lake/raw")
    p.add_argument("--submission-id", type=int, default=None)
    a = p.parse_args()
    out = Path(a.out)
    if a.submission_id:
        fetch_episode_replays_for_submission(a.submission_id, out)
    else:
        fetch_daily(a.since, a.until, out)
