"""FIELD-Zero network: deck-conditional masked pointer policy.

Design contract (from the plan):
  * Pointer/legal-option scorer — score EACH legal option, softmax over legal
    set only. No fixed action vocabulary; new deck != new output layer.
  * Deck conditioning: deck embedding = weighted sum of card embeddings +
    aggregate stats, so one brain pilots many lists.
  * Four heads: policy, value, opponent archetype (aux), hidden-resource
    probabilities (aux, becomes search priors).
  * Encoder runs ONCE per decision; the leaf evaluator used inside search is a
    tiny MLP over (root latent, cheap delta features) — see leaf_value_head.
  * Keep it SMALL. This must distill to a quantized numpy forward pass that
    runs on 2 vCPUs under a 10-minute chess clock.

GRU history encoding is optional (use_history flag) — ablate engineered belief
features vs learned memory before paying for recurrence at inference.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CARDS = 2100          # ADAPT: len(all_card_data()) + margin
CARD_EMB = 48
STATE_FEATS = 160       # ADAPT: featurizer output size
BELIEF_FEATS = 64       # ADAPT
OPTION_FEATS = 40       # ADAPT: per-option featurizer output
DECK_STATS = 16
HIDDEN = 256
LATENT = 192
N_ARCHETYPES = 24       # from build_dataset archetype clustering
N_RESOURCE_PROBS = 6    # has_evolution, has_energy, has_gust, has_switch, has_draw, lethal_next


class DeckEncoder(nn.Module):
    def __init__(self, card_emb: nn.Embedding):
        super().__init__()
        self.card_emb = card_emb
        self.proj = nn.Linear(CARD_EMB + DECK_STATS, 64)

    def forward(self, deck_ids, deck_counts, deck_stats):
        # deck_ids: (B, U) unique ids; deck_counts: (B, U); deck_stats: (B, DECK_STATS)
        emb = self.card_emb(deck_ids)                       # (B, U, E)
        w = deck_counts.unsqueeze(-1).float()
        pooled = (emb * w).sum(1) / w.sum(1).clamp(min=1.0)  # count-weighted mean
        return F.relu(self.proj(torch.cat([pooled, deck_stats], dim=-1)))


class FieldZeroNet(nn.Module):
    def __init__(self, use_history: bool = False):
        super().__init__()
        self.card_emb = nn.Embedding(N_CARDS, CARD_EMB, padding_idx=0)
        self.deck_enc = DeckEncoder(self.card_emb)
        self.use_history = use_history
        enc_in = STATE_FEATS + BELIEF_FEATS + 64
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT), nn.ReLU(),
        )
        if use_history:
            self.gru = nn.GRU(LATENT, LATENT, batch_first=True)

        # option scorer: q_i = MLP([h, E(a_i)])
        self.opt_mlp = nn.Sequential(
            nn.Linear(LATENT + OPTION_FEATS + CARD_EMB, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.archetype_head = nn.Linear(LATENT, N_ARCHETYPES)
        self.resource_head = nn.Linear(LATENT, N_RESOURCE_PROBS)
        # tiny leaf evaluator for search: (root latent, cheap delta feats) -> value
        self.leaf_value_head = nn.Sequential(
            nn.Linear(LATENT + 24, 48), nn.ReLU(), nn.Linear(48, 1), nn.Tanh(),
        )

    def encode(self, state_f, belief_f, deck_ids, deck_counts, deck_stats, h_prev=None):
        d = self.deck_enc(deck_ids, deck_counts, deck_stats)
        h = self.encoder(torch.cat([state_f, belief_f, d], dim=-1))
        if self.use_history and h_prev is not None:
            h, h_next = self.gru(h.unsqueeze(1), h_prev)
            return h.squeeze(1), h_next
        return h, None

    def score_options(self, h, opt_feats, opt_card_ids, legal_mask):
        """h: (B, L); opt_feats: (B, K, OPTION_FEATS); opt_card_ids: (B, K);
        legal_mask: (B, K) bool. Returns log-probs over the K padded slots."""
        B, K, _ = opt_feats.shape
        card_e = self.card_emb(opt_card_ids)                    # (B, K, E)
        hh = h.unsqueeze(1).expand(B, K, h.shape[-1])
        q = self.opt_mlp(torch.cat([hh, opt_feats, card_e], -1)).squeeze(-1)  # (B, K)
        q = q.masked_fill(~legal_mask, float("-inf"))
        return F.log_softmax(q, dim=-1)

    def forward(self, batch, h_prev=None):
        h, h_next = self.encode(batch["state_f"], batch["belief_f"],
                                batch["deck_ids"], batch["deck_counts"],
                                batch["deck_stats"], h_prev)
        logp = self.score_options(h, batch["opt_feats"], batch["opt_card_ids"],
                                  batch["legal_mask"])
        return {
            "log_policy": logp,
            "value": self.value_head(h).squeeze(-1),
            "archetype_logits": self.archetype_head(h),
            "resource_logits": self.resource_head(h),
            "latent": h,
            "h_next": h_next,
        }

    def leaf_value(self, root_latent, delta_feats):
        return self.leaf_value_head(torch.cat([root_latent, delta_feats], -1)).squeeze(-1)


def bc_loss(out, batch, lam_v=0.5, lam_z=0.2, lam_r=0.2):
    """Weighted behavior cloning + aux losses. batch['weight'] carries the
    skill/confidence/recency/diversity weights from the miner."""
    w = batch["weight"]
    lp = out["log_policy"].gather(1, batch["chosen_idx"].unsqueeze(1)).squeeze(1)
    l_pi = -(w * lp).sum() / w.sum()
    l_v = (w * (out["value"] - batch["outcome"]).pow(2)).sum() / w.sum()
    l_z = F.cross_entropy(out["archetype_logits"], batch["aux_archetype"],
                          reduction="none")
    l_z = (w * l_z).sum() / w.sum()
    l_r = F.binary_cross_entropy_with_logits(out["resource_logits"],
                                             batch["aux_resources"],
                                             reduction="none").mean(-1)
    l_r = (w * l_r).sum() / w.sum()
    return l_pi + lam_v * l_v + lam_z * l_z + lam_r * l_r, {
        "pi": l_pi.item(), "v": l_v.item(), "z": l_z.item(), "r": l_r.item(),
    }
