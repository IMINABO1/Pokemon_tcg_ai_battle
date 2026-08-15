"""Schema discovery. Run this on ONE real replay before touching parse_replay.py.

Prints the recursive structure (keys, types, list lengths, sample scalars) of a
replay JSON so the parser is frozen from reality, not assumptions.

Usage: python inspect_replay.py data_lake/raw/2026-07-08/12345.json [--depth 6]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def describe(node, depth, max_depth, prefix=""):
    if depth > max_depth:
        print(f"{prefix}...")
        return
    ind = "  " * depth
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                tag = f"dict({len(v)})" if isinstance(v, dict) else f"list[{len(v)}]"
                print(f"{ind}{k}: {tag}")
                # For long homogeneous lists, describe first element only.
                if isinstance(v, list) and v:
                    describe(v[0], depth + 1, max_depth, prefix=ind + "  [0] ")
                elif isinstance(v, dict):
                    describe(v, depth + 1, max_depth)
            else:
                s = repr(v)
                print(f"{ind}{k}: {type(v).__name__} = {s[:80]}")
    elif isinstance(node, list):
        print(f"{ind}{prefix}list[{len(node)}]")
        if node:
            describe(node[0], depth + 1, max_depth)
    else:
        print(f"{ind}{prefix}{type(node).__name__} = {repr(node)[:80]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--depth", type=int, default=6)
    a = p.parse_args()
    data = json.loads(Path(a.path).read_text())
    describe(data, 0, a.depth)
    # Kaggle replays usually carry a 'steps' array of per-agent
    # {observation, action, reward, status, info}. Confirm before parsing.
    if isinstance(data, dict) and "steps" in data:
        print("\n--- steps[1][0] observation keys ---")
        try:
            obs = data["steps"][1][0]["observation"]
            print(sorted(obs.keys()))
        except Exception as e:  # noqa: BLE001
            print("could not introspect steps:", e)
