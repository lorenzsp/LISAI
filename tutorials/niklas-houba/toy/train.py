"""Generate the toy datasets and train the two staged checkpoints:

    data/toy_train.npz, data/toy_val.npz   — pre-generated, seeded
    data/after_ordered.pt                  — the broken ordered-loss model
    data/after_hungarian.pt                — the permutation-invariant model

Run once before the workshop:  python workshop/toy/train.py
Both trainings take about a minute each on a laptop CPU.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from simulator import make_dataset
from model import ToySlotNet, ordered_loss, hungarian_loss

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_TRAIN, N_VAL = 8000, 1000
EPOCHS, BATCH, LR = 30, 128, 2e-3
SEED = 0


def get_data():
    train_path = os.path.join(DATA_DIR, "toy_train.npz")
    val_path = os.path.join(DATA_DIR, "toy_val.npz")
    if not os.path.exists(train_path):
        X, T = make_dataset(N_TRAIN, seed=SEED)
        np.savez_compressed(train_path, X=X, T=T)
        Xv, Tv = make_dataset(N_VAL, seed=SEED + 1)
        np.savez_compressed(val_path, X=Xv, T=Tv)
    tr, va = np.load(train_path), np.load(val_path)
    return (torch.from_numpy(tr["X"]), torch.from_numpy(tr["T"]),
            torch.from_numpy(va["X"]), torch.from_numpy(va["T"]))


def train(loss_fn, tag, X, T, Xv, Tv):
    torch.manual_seed(SEED)
    net = ToySlotNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = X.shape[0]
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(net(X[idx]), T[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        with torch.no_grad():
            # validation is ALWAYS Hungarian-matched: evaluation must respect
            # the permutation symmetry even when training does not
            val = hungarian_loss(net(Xv), Tv).item()
        history.append((epoch_loss / n, val))
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"[{tag}] epoch {epoch:3d}  train {history[-1][0]:.4f}  "
                  f"val(matched) {val:.4f}")
    dt = time.time() - t0
    torch.save({"state_dict": net.state_dict(), "history": history,
                "epochs": EPOCHS, "seed": SEED, "wall_s": dt},
               os.path.join(DATA_DIR, f"after_{tag}.pt"))
    print(f"[{tag}] done in {dt:.1f}s -> data/after_{tag}.pt")
    return net


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    X, T, Xv, Tv = get_data()
    print(f"data: train {tuple(X.shape)}, val {tuple(Xv.shape)}")
    train(ordered_loss, "ordered", X, T, Xv, Tv)
    train(hungarian_loss, "hungarian", X, T, Xv, Tv)
