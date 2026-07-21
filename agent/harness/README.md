# Local harness

Drives matches on the real engine via `cg.game` (no Kaggle runner needed).

**Critical constraint:** `Battle.battle_ptr` is a process-level singleton — exactly one
match per Python process. For many matches, use one process per match (never in-process
concurrency). `run_matches.py` (Week 2) will use `ProcessPoolExecutor` for this.

## Files
- `local_match.py` — `run_one_match(deck0, agent0, deck1, agent1)`; always calls
  `battle_finish()` in `finally`. An agent crash / illegal select becomes a forced loss.
- `smoke_match.py` — one heuristic-vs-random game; sanity check. `python harness/smoke_match.py`
- `results/` — CSV/JSONL outputs (gitignored).

## Quick start (from `agent/`)
```
python harness/smoke_match.py            # one sanity match
python -m pytest tests/ -q               # unit tests (needs dev deps)
./scripts/build_submission.sh            # build dist/submission.tar.gz
python scripts/validate_submission.py    # validate the packaged artifact in isolation
```

## macOS note
The `cg` binaries ship with a Safari quarantine xattr that macOS blocks
(`library load disallowed by system policy`). Strip it once for local dev:
`xattr -d com.apple.quarantine sample_submission/sample_submission/cg/libcg.dylib`
(and the other binaries). This is local-only — `tar` drops xattrs, so the Kaggle
Linux container is unaffected.
