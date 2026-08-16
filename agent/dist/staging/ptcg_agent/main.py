"""Thin Kaggle submission entrypoint for PTCG AI Battle.

The Kaggle runner exec()s this file with an EMPTY namespace: `__file__` is not
defined, and the LAST callable defined here becomes the agent. So: never touch
`__file__` at module level, and keep `agent` as the final definition.
"""
import os
import sys


def _bootstrap_sys_path():
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass  # exec'd by the Kaggle runner
    candidates.append("/kaggle_simulations/agent")
    candidates.append(os.getcwd())
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "ptcg_agent")) and c not in sys.path:
            sys.path.insert(0, c)


_bootstrap_sys_path()

from ptcg_agent.policy import agent_decide


def agent(obs_dict: dict) -> list[int]:
    """Chosen option indices, or the 60-card deck list when select is None."""
    return agent_decide(obs_dict)
