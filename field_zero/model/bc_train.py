"""Stage-0 behavior cloning on mined field decisions -> checkpoint BC-0.

BC-0 is the initialization for league PPO, never the final agent. Gate note:
BC validation accuracy is NOT a ladder predictor — the only graduation
criterion is the real-field gauntlet + >=2 ladder readings.

The featurizer (decision row -> tensors) is the deliberate remaining seam: it
must EXACTLY mirror the live featurizer used in the submission runtime, so
implement it once in a shared module imported by both. A train/inference
featurizer mismatch is the classic silent killer here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pointer_policy import FieldZeroNet, bc_loss

MAX_OPTIONS = 48  # pad/truncate legal option lists; log truncation rate


class DecisionDataset(Dataset):
    def __init__(self, parquet: Path, split: str, featurizer):
        df = pd.read_parquet(parquet)
        df = df[df["split"] == split]
        # drop rows whose chosen action fell outside the padded option window
        self.rows = df.reset_index(drop=True)
        self.featurizer = featurizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        obs = json.loads(r["observation"])
        belief = json.loads(r["belief_features"])
        legal = json.loads(r["legal_options"])[:MAX_OPTIONS]
        chosen = json.loads(r["chosen_indices"])
        deck = json.loads(r["own_deck_ids"]) if r["own_deck_ids"] else []
        return self.featurizer(
            obs=obs, belief=belief, legal=legal, chosen=chosen, deck=deck,
            outcome=r["outcome"], weight=r["weight"],
            aux_archetype=r["aux_opponent_archetype"],
        )


def train(decisions: Path, featurizer, epochs: int, lr: float, out: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = FieldZeroNet(use_history=False).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    tr = DataLoader(DecisionDataset(decisions, "train", featurizer),
                    batch_size=512, shuffle=True, num_workers=4)
    va = DataLoader(DecisionDataset(decisions, "val", featurizer),
                    batch_size=512, num_workers=2)

    for ep in range(epochs):
        net.train()
        for batch in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            out_ = net(batch)
            loss, parts = bc_loss(out_, batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        # ---- val: top-1 imitation accuracy on FUTURE games ----
        net.eval(); correct = n = 0
        with torch.no_grad():
            for batch in va:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = net(batch)["log_policy"].argmax(-1)
                correct += (pred == batch["chosen_idx"]).sum().item()
                n += len(pred)
        acc = correct / max(n, 1)
        print(f"epoch {ep}: val_top1={acc:.4f} {parts}")
        torch.save(net.state_dict(), out / f"bc_ep{ep}.pt")
    torch.save(net.state_dict(), out / "BC-0.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="data_lake/decisions.parquet")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    # featurizer: import the SHARED live/train featurizer here once written.
    raise SystemExit(
        "Wire the shared featurizer (obs->tensors) before running; it must be "
        "the same module the submission runtime uses."
    )
