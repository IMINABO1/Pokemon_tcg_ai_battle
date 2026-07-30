"""Submission builder and packaging script for Kaggle."""
import os
from pathlib import Path
import shutil
import sys
import tarfile

# Ensure src is in sys.path for legality check
root_dir = Path(__file__).resolve().parent.parent.parent
src_dir = root_dir / "agent" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ptcg_agent.carddb import validate_deck_legality


def build_submission():
    agent_dir = root_dir / "agent"
    dist_dir = agent_dir / "dist"
    staging_dir = dist_dir / "staging"
    tar_path = dist_dir / "submission.tar.gz"

    # 1. Clean staging directory
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check deck legality - HARD FAIL if illegal
    deck_csv_path = agent_dir / "decks" / "candidate_v1.csv"
    if not deck_csv_path.exists():
        deck_csv_path = src_dir / "ptcg_agent" / "deck.csv"

    with open(deck_csv_path, "r", encoding="utf-8") as f:
        deck_ids = [int(line.strip()) for line in f.read().split("\n") if line.strip()]

    is_legal, reason = validate_deck_legality(deck_ids)
    if not is_legal:
        print(f"BUILD FAILED: Deck is illegal - {reason}")
        sys.exit(1)
    print(f"Deck legality check passed: {reason}")

    # 3. Copy top-level main.py and deck.csv
    shutil.copy(src_dir / "ptcg_agent" / "main.py", staging_dir / "main.py")
    shutil.copy(deck_csv_path, staging_dir / "deck.csv")

    # 4. Copy ptcg_agent package (excluding dev-only logging_utils.py)
    pkg_target = staging_dir / "ptcg_agent"
    pkg_target.mkdir(parents=True, exist_ok=True)

    for item in (src_dir / "ptcg_agent").glob("*.py"):
        if item.name != "logging_utils.py":
            shutil.copy(item, pkg_target / item.name)
    shutil.copy(deck_csv_path, pkg_target / "deck.csv")

    # 5. Copy cg directory from sample_submission
    cg_source = root_dir / "sample_submission" / "sample_submission" / "cg"
    cg_target = staging_dir / "cg"
    if cg_target.exists():
        shutil.rmtree(cg_target)
    shutil.copytree(cg_source, cg_target)

    # 6. Build submission.tar.gz
    if tar_path.exists():
        tar_path.unlink()

    with tarfile.open(tar_path, "w:gz") as tar:
        for file_path in staging_dir.rglob("*"):
            arcname = file_path.relative_to(staging_dir)
            tar.add(file_path, arcname=arcname)

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"Submission package created at {tar_path} ({size_mb:.2f} MB)")

    if size_mb > 197.7:
        print(f"BUILD FAILED: Package size ({size_mb:.2f} MB) exceeds 197.7 MB limit.")
        sys.exit(1)

    print("BUILD SUCCESSFUL!")


if __name__ == "__main__":
    build_submission()
