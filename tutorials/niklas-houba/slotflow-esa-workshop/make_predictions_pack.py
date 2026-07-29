"""Precompute everything Tutorial 2 needs from the pretrained SlotFlow model.

Run once, offline, before the workshop (needs the release checkpoint in
pretrained_model/test_clariden/checkpoints/):

    python workshop/make_predictions_pack.py

Output: workshop/predictions_pack.npz with three groups —

  nominal_*    1200 in-distribution signals (seed 42): the training regime.
  stress_*     800 out-of-distribution signals (seed 4242): crowded
               frequencies (min separation 0.004 Hz, below the training
               floor of 0.01 Hz) and loud noise (sigma 1.0-2.0). This is
               where the failure gallery lives.
  stab_*       one fixed K=5 catalogue x 20 fresh noise realizations,
               for the slot-identity stability experiment.

Per signal: |FFT| spectrum of the long stream (float16, for plotting),
cardinality posterior q(K|x), predicted/true K, true parameters, and for
ALL 10 forced slots: MAP parameters, spreads, 64 posterior samples
(float16), plus the 512-d flow embedding. Slot i's context is
[g, onehot(i)] regardless of how many slots are instantiated, so
forced-10 inference yields the per-slot posteriors of every prefix.
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import MultiSinusoidDataset  # noqa: E402
from src.model import SlotFlow  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
M = 10                 # max slots
N_SAMPLES = 128        # drawn per slot
N_KEEP = 64            # stored per slot (float16)
BATCH = 25
FREQ_RANGE = (2.5, 3.0)
FREQ_WEIGHT = 3.0      # training-time scaling of the frequency coordinate
F_MID = sum(FREQ_RANGE) / 2

SMOKE = "--smoke" in sys.argv  # tiny run for timing/sanity


def load_model():
    cfg = torch.load(os.path.join(ROOT, "pretrained_model/test_clariden/model_config.pt"),
                     map_location="cpu", weights_only=False)
    ckpt_dir = os.path.join(ROOT, "pretrained_model/test_clariden/checkpoints")
    ckpt_file = sorted(f for f in os.listdir(ckpt_dir)
                       if f.endswith(".ckpt") and ("best" in f or "hpc_ckpt" in f))[-1]
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location="cpu",
                      weights_only=False)
    state = {k.replace("model.", "").replace("_orig_mod.", ""): v
             for k, v in ckpt["state_dict"].items()}
    model = SlotFlow(hidden_dim=cfg["hidden_dim"], max_slots=cfg["max_slots"],
                     use_noise_encoder=cfg.get("use_noise_encoder", False))
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model, cfg


def invert_params(s):
    """(..., 4) flow coords [amp, cos, sin, scaled freq] -> (..., 3) [amp, phase, freq]."""
    amp = s[..., 0]
    phase = torch.atan2(s[..., 2], s[..., 1]) % (2 * math.pi)
    freq = s[..., 3] / FREQ_WEIGHT + F_MID
    return torch.stack([amp, phase, freq], dim=-1)


def circular_stats(phases, dim):
    c, s = torch.cos(phases).mean(dim), torch.sin(phases).mean(dim)
    mean = torch.atan2(s, c) % (2 * math.pi)
    R = torch.sqrt(c**2 + s**2).clamp(1e-6, 1.0)
    std = torch.sqrt(-2 * torch.log(R))
    return mean, std


@torch.no_grad()
def process(model, x_long, x_short):
    """One batch -> dict of numpy arrays."""
    B = x_long.shape[0]
    out_pred = model(x_long, x_short, use_gt_k=None)
    q_k = F.softmax(out_pred["K_logits"], dim=-1)                    # (B, M)
    k_pred = out_pred["K_pred"]                                      # (B,)
    out_all = model(x_long, x_short,
                    use_gt_k=torch.full((B,), M, dtype=torch.long, device=DEVICE))
    ctx = out_all["context"]                                         # (B*M, ctx)
    samp = model.flow.sample(N_SAMPLES, context=ctx)                 # (B*M, S, 4)
    samp = invert_params(samp).view(B, M, N_SAMPLES, 3)              # (B, M, S, 3)

    map_amp = samp[..., 0].median(dim=-1).values
    map_freq = samp[..., 2].median(dim=-1).values
    map_phase, std_phase = circular_stats(samp[..., 1], dim=-1)
    maps = torch.stack([map_amp, map_phase, map_freq], dim=-1)       # (B, M, 3)
    stds = torch.stack([samp[..., 0].std(dim=-1), std_phase,
                        samp[..., 2].std(dim=-1)], dim=-1)

    spec = torch.fft.rfft(x_long, dim=-1, norm="ortho").abs()        # (B, Lr)
    return dict(
        q_k=q_k.cpu().numpy(),
        k_pred=k_pred.cpu().numpy().astype(np.int16),
        maps=maps.cpu().numpy().astype(np.float32),
        stds=stds.cpu().numpy().astype(np.float32),
        # float32 is load-bearing: the freq posteriors are ~2e-4 Hz wide at
        # f~2.7 Hz, an order of magnitude below float16 resolution there —
        # float16 storage collapses the samples and breaks interval coverage.
        samples=samp[:, :, :N_KEEP].cpu().numpy().astype(np.float32),
        embed=out_all["h_embed_flow"].cpu().numpy().astype(np.float16),
        spec=spec.cpu().numpy().astype(np.float16),
    )


def run_subset(model, cfg, name, n, seed, min_sep, noise, allowed_k):
    ds = MultiSinusoidDataset(
        set_size=n,
        num_samples_long=cfg["num_samples_long"], tEnd_long=cfg["tEnd_long"],
        num_samples_short=cfg["num_samples_short"], tEnd_short=cfg["tEnd_short"],
        max_components=cfg["max_components"],
        freq_range=cfg["freq_range"], amp_range=cfg["amp_range"],
        min_freq_sep=min_sep, noise_std=noise, seed=seed,
        mode="inference", allowed_K_values=allowed_k,
    )
    outs, k_true, truths = [], [], []
    t0 = time.time()
    for start in range(0, n, BATCH):
        idx = range(start, min(start + BATCH, n))
        items = [ds[i] for i in idx]
        x_long = torch.stack([it[0] for it in items]).to(DEVICE)
        x_short = torch.stack([it[1] for it in items]).to(DEVICE)
        outs.append(process(model, x_long, x_short))
        for it in items:
            k = int(it[2])
            k_true.append(k)
            pad = np.zeros((M, 3), np.float32)
            pad[:k] = np.asarray(it[4][:k], np.float32)  # (amp, phase, freq)
            truths.append(pad)
        done = start + len(items)
        print(f"[{name}] {done}/{n}  ({(time.time()-t0)/done:.2f}s/signal)",
              flush=True)
    merged = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    merged["k_true"] = np.asarray(k_true, np.int16)
    merged["truth"] = np.stack(truths)
    return {f"{name}_{k}": v for k, v in merged.items()}


def _stab_catalogue(seed=7, K=5):
    """The fixed stability catalogue (deterministic given seed)."""
    rng = np.random.default_rng(seed)
    freqs = np.sort(rng.uniform(2.55, 2.95, K))
    while np.diff(freqs).min() < 0.02:  # moderately separated, unambiguous
        freqs = np.sort(rng.uniform(2.55, 2.95, K))
    amps = rng.uniform(0.7, 1.3, K)
    phis = rng.uniform(0, 2 * math.pi, K)
    return amps, phis, freqs


def _render_and_process(model, cfg, amps, phis, freqs, n_real, gen_seed,
                        noise_scale=1.0):
    """Render a fixed catalogue under n_real fresh noise draws and run the
    model. Returns process() output + the truth array."""
    t_long = torch.linspace(0, cfg["tEnd_long"], cfg["num_samples_long"])
    t_short = torch.linspace(0, cfg["tEnd_short"], cfg["num_samples_short"])
    n_master = int(round(cfg["tEnd_long"] /
                         (cfg["tEnd_short"] / cfg["num_samples_short"])))
    A = torch.tensor(amps, dtype=torch.float32)[:, None]
    f = torch.tensor(freqs, dtype=torch.float32)[:, None]
    p = torch.tensor(phis, dtype=torch.float32)[:, None]
    sig_long = (A * torch.sin(2 * math.pi * f * t_long[None] + p)).sum(0)
    sig_short = (A * torch.sin(2 * math.pi * f * t_short[None] + p)).sum(0)

    g = torch.Generator().manual_seed(gen_seed)
    xl, xs = [], []
    for _ in range(n_real):
        nm = torch.randn(n_master, generator=g) * noise_scale
        nl = F.interpolate(nm[None, None], size=len(t_long), mode="linear",
                           align_corners=False).squeeze()
        ns = F.interpolate(nm[None, None], size=len(t_short), mode="linear",
                           align_corners=False).squeeze()
        xl.append(sig_long + nl)
        xs.append(sig_short + ns)
    out = process(model, torch.stack(xl).to(DEVICE), torch.stack(xs).to(DEVICE))
    out["truth"] = np.stack([amps, phis, freqs], axis=-1).astype(np.float32)
    return out


def run_structural(model, cfg, n_real=8):
    """stab2: the stability catalogue under STRUCTURAL perturbations.

    base    the fixed K=5 catalogue (same as the stab group)
    insert  an extra source added BELOW every existing frequency (f=2.50)
    remove  the middle-frequency source deleted (K=4)
    shift   every frequency shifted by +10 mHz (adjacent-window proxy)

    Question the notebook asks: when the catalogue itself changes, does
    slot i keep tracking "its" source from the base configuration?
    """
    amps, phis, freqs = _stab_catalogue()
    rng = np.random.default_rng(99)
    configs = {
        "base": (amps, phis, freqs),
        "insert": (np.append(amps, 1.0), np.append(phis, rng.uniform(0, 2 * math.pi)),
                   np.append(freqs, 2.50)),
        "remove": (np.delete(amps, 2), np.delete(phis, 2), np.delete(freqs, 2)),
        "shift": (amps, phis, freqs + 0.010),
    }
    out = {}
    for name, (a, p, f) in configs.items():
        r = _render_and_process(model, cfg, a, p, f, n_real,
                                gen_seed=100 + len(name))
        for k in ["q_k", "k_pred", "maps", "stds", "truth"]:
            out[f"stab2_{name}_{k}"] = r[k]
    return out


def run_stability(model, cfg, n_real=20, seed=7, noise_scale=1.0):
    """One fixed K=5 catalogue, n_real fresh noise realizations."""
    rng = np.random.default_rng(seed)
    K = 5
    freqs = np.sort(rng.uniform(2.55, 2.95, K))
    while np.diff(freqs).min() < 0.02:  # moderately separated, unambiguous
        freqs = np.sort(rng.uniform(2.55, 2.95, K))
    amps = rng.uniform(0.7, 1.3, K)
    phis = rng.uniform(0, 2 * math.pi, K)

    t_long = torch.linspace(0, cfg["tEnd_long"], cfg["num_samples_long"])
    t_short = torch.linspace(0, cfg["tEnd_short"], cfg["num_samples_short"])
    n_master = int(round(cfg["tEnd_long"] / (cfg["tEnd_short"] / cfg["num_samples_short"])))
    A = torch.tensor(amps, dtype=torch.float32)[:, None]
    f = torch.tensor(freqs, dtype=torch.float32)[:, None]
    p = torch.tensor(phis, dtype=torch.float32)[:, None]
    sig_long = (A * torch.sin(2 * math.pi * f * t_long[None] + p)).sum(0)
    sig_short = (A * torch.sin(2 * math.pi * f * t_short[None] + p)).sum(0)

    g = torch.Generator().manual_seed(seed)
    xl, xs = [], []
    for _ in range(n_real):
        nm = torch.randn(n_master, generator=g) * noise_scale
        nl = F.interpolate(nm[None, None], size=len(t_long), mode="linear",
                           align_corners=False).squeeze()
        ns = F.interpolate(nm[None, None], size=len(t_short), mode="linear",
                           align_corners=False).squeeze()
        xl.append(sig_long + nl)
        xs.append(sig_short + ns)
    out = process(model, torch.stack(xl).to(DEVICE), torch.stack(xs).to(DEVICE))
    truth = np.stack([amps, phis, freqs], axis=-1).astype(np.float32)  # (K, 3)
    out["truth"] = truth
    return {f"stab_{k}": v for k, v in out.items()}


if __name__ == "__main__":
    model, cfg = load_model()
    print(f"model loaded on {DEVICE}; smoke={SMOKE}", flush=True)

    if "--stab2-merge" in sys.argv:
        # incremental: add/refresh the stab2 group in the existing pack
        out_path = os.path.join(ROOT, "predictions_pack.npz")
        existing = dict(np.load(out_path))
        existing.update(run_structural(model, cfg))
        np.savez_compressed(out_path, **existing)
        print(f"merged stab2 into {out_path} "
              f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
        sys.exit(0)

    n_nom, n_str = (8, 8) if SMOKE else (1200, 800)
    pack = {}
    pack.update(run_subset(model, cfg, "nominal", n_nom, seed=42,
                           min_sep=0.01, noise=cfg["noise_std"],
                           allowed_k=list(range(1, M + 1))))
    pack.update(run_subset(model, cfg, "stress", n_str, seed=4242,
                           min_sep=0.004, noise=(1.0, 2.0),
                           allowed_k=list(range(5, M + 1))))
    pack.update(run_stability(model, cfg))
    pack.update(run_structural(model, cfg))
    out_path = os.path.join(ROOT,
                        "predictions_pack_smoke.npz" if SMOKE
                        else "predictions_pack.npz")
    np.savez_compressed(out_path, **pack)
    size = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path} ({size:.1f} MB)")
