"""Thin Kaggle submission entrypoint for PTCG AI Battle."""
import os
import sys
from pathlib import Path

# Ensure current working directory / package root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
cwd = Path.cwd()
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

from ptcg_agent.policy import agent_decide, read_deck_csv


def agent(obs_dict: dict) -> list[int]:
    """Top-level agent function called by Kaggle competition runner.
    
    Returns:
        list[int]: Chosen option indices or 60-card deck list when select is None.
    """
    return agent_decide(obs_dict)
