"""Packaged submission artifact validation script."""
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

root_dir = Path(__file__).resolve().parent.parent.parent
tar_path = root_dir / "agent" / "dist" / "submission.tar.gz"


def validate_submission():
    if not tar_path.exists():
        print(f"Validation FAILED: Archive {tar_path} does not exist. Run build_submission.py first.")
        sys.exit(1)

    print(f"Validating submission archive: {tar_path}...")

    # Extract to temp directory
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_path)


        # 1. Verify structure
        main_py = temp_path / "main.py"
        deck_csv = temp_path / "deck.csv"
        cg_dir = temp_path / "cg"
        agent_dir = temp_path / "ptcg_agent"

        assert main_py.exists(), "Missing main.py at top level of tar archive!"
        assert deck_csv.exists(), "Missing deck.csv at top level of tar archive!"
        assert cg_dir.exists(), "Missing cg directory!"
        assert agent_dir.exists(), "Missing ptcg_agent directory!"

        print("  Structure check PASSED")

        # 2. Check deck.csv content
        with open(deck_csv, "r") as f:
            deck_ids = [int(line.strip()) for line in f.read().split("\n") if line.strip()]
        assert len(deck_ids) == 60, f"Expected 60 cards in deck.csv, got {len(deck_ids)}"
        print("  Deck CSV check PASSED (60 card IDs)")

        # 3. Test running packaged main.py in isolated process
        if str(root_dir / "agent") not in sys.path:
            sys.path.insert(0, str(root_dir / "agent"))
        sys.path.insert(0, str(temp_path))
        import main
        from harness.local_match import run_one_match, random_agent


        # Sample deck for opponent
        sample_deck_file = root_dir / "sample_submission" / "sample_submission" / "deck.csv"
        with open(sample_deck_file, "r") as f:
            sample_deck = [int(line.strip()) for line in f.read().split("\n") if line.strip()]

        print("  Running packaged artifact test match...")
        res = run_one_match(
            deck0=deck_ids,
            agent0=main.agent,
            name0="Packaged_Agent",
            deck1=sample_deck,
            agent1=random_agent,
            name1="Random_Agent",
            max_actions=500
        )

        assert res.error_message is None, f"Packaged agent test match failed: {res.error_message}"
        print(f"  Packaged match completed: Winner = {res.winner_name} in {res.total_actions} actions ({res.duration_seconds:.2f}s)")

    print("\nALL SUBMISSION VALIDATION CHECKS PASSED!")


if __name__ == "__main__":
    validate_submission()
