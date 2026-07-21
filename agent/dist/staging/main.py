"""Kaggle entrypoint. Thin wrapper — all logic lives in ptcg_agent.policy.

Kaggle's runner imports this module and calls `agent(obs_dict)`. Keep it stable and
minimal so internals can iterate without touching the competition-facing surface.
"""
from ptcg_agent.policy import agent_decide, read_deck_csv  # noqa: F401


def agent(obs_dict: dict) -> list[int]:
    return agent_decide(obs_dict)
