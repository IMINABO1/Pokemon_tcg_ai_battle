"""Stage-1: masked-pointer league PPO from BC-0 (implements the doc-31 recipe).

Pipeline position:  BC-0  ->  [THIS]  ->  adversarial mining  ->  distill.

What's implemented exactly as specced:
  * Clipped PPO with GAE over the SAME pointer distribution BC trained
    (softmax over legal options only — variable action spaces are native).
  * League opponent sampling: current-main / recent / old-strong /
    heuristic+PIMC / exploiters / field decks, with prioritized reweighting
    toward opponents where P(win) ~ 0.5 (exp(-|WR-0.5|/tau)).
  * BC anchor: L = L_PPO + lam_bc * L_BC, lam_bc decaying over training
    (keeps a small floor to resist catastrophic strategy drift; the floor
    itself is an experiment — set BC_FLOOR=0 to ablate).
  * Terminal-only reward (+1/-1/0) with OPTIONAL potential-based shaping
    r' = r + gamma*Phi(s') - Phi(s), where Phi is the existing heuristic
    evaluate_state — policy-invariant under standard conditions, so it
    densifies learning without teaching "damage for damage's sake".

The environment adapter (self-play worker driving cg.game battles and
emitting featurized trajectories) is the deliberate remaining seam — it
reuses harness/local_match.py mechanics. Workers are separate OS processes
(Battle is a per-process singleton; engine RNG is shared per tree).
"""
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from pointer_policy import FieldZeroNet

GAMMA = 1.0            # episodic terminal reward; horizon fits in one game
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
BC_START, BC_FLOOR, BC_DECAY_STEPS = 0.5, 0.02, 200_000
LEAGUE_TAU = 0.15      # prioritized-opponent temperature

LEAGUE_BASE_MIX = {    # doc-31 base mixture; prioritization reweights within groups
    "current_main": 0.35,
    "recent_ckpts": 0.20,
    "old_strong": 0.15,
    "scripted": 0.10,   # heuristic + PIMC baselines
    "exploiters": 0.10,
    "field_decks": 0.10,
}


@dataclass
class LeagueMember:
    name: str
    group: str
    policy_fn: object                 # callable(obs)->action  (frozen)
    wins: int = 0
    games: int = 0

    @property
    def wr(self) -> float:
        return self.wins / self.games if self.games else 0.5


@dataclass
class League:
    members: list[LeagueMember] = field(default_factory=list)

    def sample_opponent(self) -> LeagueMember:
        groups: dict[str, list[LeagueMember]] = {}
        for m in self.members:
            groups.setdefault(m.group, []).append(m)
        gnames = [g for g in LEAGUE_BASE_MIX if g in groups]
        gw = torch.tensor([LEAGUE_BASE_MIX[g] for g in gnames])
        g = gnames[torch.multinomial(gw / gw.sum(), 1).item()]
        cands = groups[g]
        # prioritize opponents near 50% winrate (from OUR perspective)
        pri = torch.tensor([float(torch.exp(torch.tensor(-abs(m.wr - 0.5) / LEAGUE_TAU)))
                            for m in cands])
        return cands[torch.multinomial(pri / pri.sum(), 1).item()]

    def record(self, member: LeagueMember, we_won: bool):
        member.games += 1
        member.wins += int(not we_won)  # member's wins = our losses


def compute_gae(rewards, values, dones):
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nonterm = 1.0 - dones[t]
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * next_v * nonterm - values[t]
        last = delta + GAMMA * GAE_LAMBDA * nonterm * last
        adv[t] = last
    return adv, adv + values[: len(rewards)]


def shaped_rewards(raw_terminal, phis, gamma=GAMMA, use_shaping=True):
    """raw_terminal: (T,) zeros except last (+1/-1/0). phis: (T+1,) heuristic
    evaluate_state potentials (phi[T] = 0 at terminal)."""
    if not use_shaping:
        return raw_terminal
    return raw_terminal + gamma * phis[1:] - phis[:-1]


def bc_lambda(step: int) -> float:
    frac = min(step / BC_DECAY_STEPS, 1.0)
    return BC_FLOOR + (BC_START - BC_FLOOR) * (1.0 - frac)


def ppo_update(net: FieldZeroNet, bc_ref: FieldZeroNet, opt, rollout, step,
               epochs=3, minibatch=512):
    """rollout: dict of tensors — batch fields (as in bc_loss) plus
    old_logp (T,), advantages (T,), returns (T,)."""
    T = rollout["chosen_idx"].shape[0]
    adv = rollout["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    lam_bc = bc_lambda(step)
    stats = {}
    for _ in range(epochs):
        for idx in torch.randperm(T).split(minibatch):
            b = {k: v[idx] for k, v in rollout.items()}
            out = net(b)
            logp = out["log_policy"].gather(1, b["chosen_idx"].unsqueeze(1)).squeeze(1)
            ratio = (logp - b["old_logp"]).exp()
            a = adv[idx]
            l_clip = -torch.min(ratio * a,
                                ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * a).mean()
            l_v = F.mse_loss(out["value"], b["returns"])
            entropy = -(out["log_policy"].exp() * out["log_policy"]
                        ).nan_to_num().sum(-1).mean()
            # BC anchor toward the frozen field-imitation reference
            with torch.no_grad():
                ref_logp = bc_ref(b)["log_policy"]
            l_bc = F.kl_div(out["log_policy"], ref_logp,
                            log_target=True, reduction="batchmean")
            loss = l_clip + VALUE_COEF * l_v - ENTROPY_COEF * entropy + lam_bc * l_bc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats = {"clip": l_clip.item(), "v": l_v.item(),
                     "ent": entropy.item(), "bc": l_bc.item(), "lam_bc": lam_bc}
    return stats


class CheckpointBuffer:
    """Feeds recent_ckpts / old_strong league groups."""
    def __init__(self, every=10_000, keep_recent=3, keep_old=4):
        self.every, self.recent = every, deque(maxlen=keep_recent)
        self.old, self.keep_old = [], keep_old

    def maybe_snapshot(self, net, step, gauntlet_wr=None):
        if step % self.every:
            return
        sd = {k: v.clone().cpu() for k, v in net.state_dict().items()}
        self.recent.append((step, sd))
        if gauntlet_wr is not None and gauntlet_wr > 0.55:
            self.old.append((step, sd, gauntlet_wr))
            self.old = sorted(self.old, key=lambda x: -x[2])[: self.keep_old]
