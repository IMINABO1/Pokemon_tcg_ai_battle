"""Structured logging utility for match harness and debugging."""
import json
import os
from pathlib import Path
from typing import Any, Optional

LOG_ENABLED = os.environ.get("PTCG_LOG_ENABLED", "0") == "1"
LOG_FILE_PATH = os.environ.get("PTCG_LOG_FILE", "")


def log_decision(
    state_turn: int,
    your_index: int,
    select_type: str,
    action_chosen: list[int],
    score: float,
    extra: Optional[dict[str, Any]] = None
):
    if not LOG_ENABLED or not LOG_FILE_PATH:
        return

    record = {
        "schema_version": 1,
        "turn": state_turn,
        "your_index": your_index,
        "select_type": select_type,
        "action_chosen": action_chosen,
        "score": score,
        "extra": extra or {},
    }

    try:
        log_path = Path(LOG_FILE_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
