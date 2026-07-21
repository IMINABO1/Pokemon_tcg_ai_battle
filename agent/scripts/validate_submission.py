"""Validate the PACKAGED artifact in isolation (proxy for the Kaggle runtime).

Extracts dist/submission.tar.gz into a temp dir and, from *inside* that dir with a
minimal sys.path (only the extracted root — stdlib + bundled cg, NO dev src/), it:
  - imports the packaged main.py and calls read_deck_csv() from cwd,
  - drives at least one full local match through packaged `agent()`,
  - asserts zero uncaught exceptions and that the fallback path itself never fails,
  - exercises edge selects (minCount==0) by construction over a full game,
  - times decisions against PER_DECISION_BUDGET_SECONDS as a proxy check.

Run from agent/:  python scripts/validate_submission.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARBALL = os.path.join(AGENT_DIR, "dist", "submission.tar.gz")

# This driver script runs INSIDE the extracted dir as a subprocess with a clean cwd
# and only the extracted root on sys.path.
_INNER = r'''
import os, sys, random, time, traceback
sys.path.insert(0, os.getcwd())          # extracted root only
import main                               # packaged main.py (top level)
from cg.api import to_observation_class
from ptcg_agent import config

# deck load from cwd
deck = main.read_deck_csv()
assert len(deck) == 60, f"deck not 60: {len(deck)}"

from cg import game
def rnd(obs_dict):
    o = to_observation_class(obs_dict)
    if o.select is None: return main.read_deck_csv()
    s = o.select
    k = s.minCount if s.minCount > 0 else min(s.maxCount, 1)
    k = min(k, len(s.option))
    return random.sample(range(len(s.option)), k) if k > 0 else []

random.seed(7)
obs, sd = game.battle_start(deck, deck)
assert obs is not None, f"battle_start failed: {sd.errorPlayer}/{sd.errorType}"
saw_optional = False
max_dec = 0.0
actions = 0
try:
    while actions < 3000:
        o = to_observation_class(obs)
        if o.current is not None and o.current.result != -1:
            break
        if o.select is None:
            break
        if o.select.minCount == 0:
            saw_optional = True
        player = o.current.yourIndex if o.current is not None else 0
        t0 = time.monotonic()
        sel = main.agent(obs) if player == 0 else rnd(obs)
        max_dec = max(max_dec, time.monotonic() - t0)
        obs = game.battle_select(sel)
        actions += 1
finally:
    game.battle_finish()

# fallback path must never fail: call it with a hand-built minimal select
from ptcg_agent import policy
class _O:  # minimal duck-typed select
    pass
fake_select = _O(); fake_select.minCount = 0; fake_select.maxCount = 1; fake_select.option = []
fake_obs = _O(); fake_obs.select = fake_select
assert policy._fallback_decide(fake_obs) == [], "fallback with minCount 0 should be []"

print(f"OK packaged actions={actions} max_dec={max_dec:.4f} budget={config.PER_DECISION_BUDGET_SECONDS} saw_optional={saw_optional}")
if max_dec > config.PER_DECISION_BUDGET_SECONDS:
    print("WARN: a decision exceeded PER_DECISION_BUDGET_SECONDS")
'''


def main() -> int:
    if not os.path.exists(TARBALL):
        print(f"ERROR: {TARBALL} not found — run scripts/build_submission.sh first")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(TARBALL) as t:
            t.extractall(tmp)
        # sanity: expected top-level files present
        for req in ("main.py", "deck.csv", "cg", "ptcg_agent"):
            if not os.path.exists(os.path.join(tmp, req)):
                print(f"ERROR: packaged artifact missing {req}")
                return 1
        # dev-only file must be absent
        if os.path.exists(os.path.join(tmp, "ptcg_agent", "logging_utils.py")):
            print("ERROR: dev-only logging_utils.py leaked into package")
            return 1

        # run the inner driver with a CLEAN environment (no dev PYTHONPATH)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", _INNER],
            cwd=tmp, env=env, capture_output=True, text=True,
        )
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print("VALIDATION FAILED:\n", proc.stderr)
            return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
