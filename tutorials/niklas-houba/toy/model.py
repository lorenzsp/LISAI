"""Tiny slot-based set predictor for the toy spectral-source problem,
plus the two training objectives compared in Tutorial 1.

    x -> 1-D CNN encoder -> MLP -> [(mu_1, A_1), (mu_2, A_2), ...]

Point predictions only — no flows. The permutation machinery (Hungarian
matching on a per-pair cost matrix) is the same idea SlotFlow uses with a
negative-log-likelihood cost.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class ToySlotNet(nn.Module):
    def __init__(self, n_bins=128, n_slots=2, hidden=128):
        super().__init__()
        self.n_slots = n_slots
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.GELU(),
        )
        enc_len = n_bins // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * enc_len, hidden), nn.GELU(),
            nn.Linear(hidden, n_slots * 2),
        )

    def forward(self, x):
        h = self.encoder(x.unsqueeze(1))
        return self.head(h).view(-1, self.n_slots, 2)  # (B, n_slots, [mu, A])


def ordered_loss(pred, target):
    """Compare slot i to target i, in whatever order the targets arrived.
    This is the deliberately broken objective."""
    return ((pred - target) ** 2).sum(dim=(1, 2)).mean()


def pairwise_cost(pred, target, w_mu=1.0, w_a=1.0):
    """Cost matrix C[b, i, j] = w_mu*(mu_hat_i - mu_j)^2 + w_a*(A_hat_i - A_j)^2."""
    d = pred.unsqueeze(2) - target.unsqueeze(1)          # (B, slots, sources, 2)
    return w_mu * d[..., 0] ** 2 + w_a * d[..., 1] ** 2  # (B, slots, sources)


def hungarian_loss(pred, target, w_mu=1.0, w_a=1.0):
    """Minimum-cost one-to-one assignment between predicted slots and true
    sources. The assignment is found on a detached copy (not differentiable);
    the loss gathers the matched entries of the live cost tensor, so
    gradients flow through the matched pairs only."""
    cost = pairwise_cost(pred, target, w_mu, w_a)        # (B, K, K)
    total = 0.0
    for b in range(cost.shape[0]):
        row, col = linear_sum_assignment(cost[b].detach().cpu().numpy())
        total = total + cost[b, row, col].sum()
    return total / cost.shape[0]
