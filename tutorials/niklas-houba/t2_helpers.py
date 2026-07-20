"""Tutorial 2 plumbing: caches and inspection helpers.

Lives in a module so the notebook stays readable — nothing in here is
conceptually load-bearing for the session. Call init() once (done by the
setup cell), then import what you need. Conventions:

 * parameter rows are [amp, phase, freq] — frequency is COLUMN 2;
 * frequencies in Hz; errors/widths/tolerances displayed in mHz;
 * K_true = injected count, K_MAP = argmax q(K|x), K_tau = #{i: p_i > tau};
 * active slots are always the K_tau prefix (slots exist in index order).
"""

import json
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from viz import C, mark_pred, title2, fmt_mhz
from catalogue_metrics import (existence_probs, match_catalogue,
                               catalogue_scores, TOL_F)

__all__ = ["pack", "gallery", "M", "SUBSETS", "EX", "TRUTHS", "FREQS_AXIS",
           "BAND", "FB", "DF_BIN", "PEAKS", "DEV", "FULL", "k_tau",
           "evaluate", "baseline_catalogue", "show_signal", "slot_table",
           "live_demo"]

pack = gallery = None
M = 10
SUBSETS = ["nominal", "stress"]
EX = TRUTHS = PEAKS = None
FREQS_AXIS = np.fft.rfftfreq(3000, d=0.1)      # long stream: 10 Hz, 300 s
BAND = (FREQS_AXIS > 2.45) & (FREQS_AXIS < 3.05)
FB = FREQS_AXIS[BAND]
DF_BIN = FB[1] - FB[0]                          # 3.33 mHz Rayleigh bin
DEV = np.arange(150)                            # public development split
FULL = None


def init(pack_path="predictions_pack.npz", gallery_path="gallery.json"):
    """Load the pack into RAM and build every cache the cells reuse.

    (An NpzFile decompresses the WHOLE array on every [] access, which
    turns per-signal loops into minutes — hence the dict.)
    """
    global pack, gallery, EX, TRUTHS, PEAKS, FULL
    t0 = time.time()
    with np.load(pack_path) as npz:
        pack = {k: npz[k] for k in npz.files}
    with open(gallery_path) as fh:
        gallery = json.load(fh)
    EX = {s: existence_probs(pack[f"{s}_q_k"]) for s in SUBSETS}
    TRUTHS = {s: [pack[f"{s}_truth"][i][:int(pack[f"{s}_k_true"][i])]
                  for i in range(len(pack[f"{s}_k_true"]))] for s in SUBSETS}
    PEAKS = {}
    for i in range(len(pack["stress_k_true"])):
        s = pack["stress_spec"][i][BAND].astype(np.float32)
        pk, props = find_peaks(s, height=0.10 * s.max(), distance=2)
        PEAKS[i] = (FB[pk], props["peak_heights"] / s.max(), pk)
    FULL = np.arange(len(pack["stress_k_true"]))
    print(f"pack + caches ready in {time.time() - t0:.1f} s — "
          f"nominal {len(pack['nominal_k_true'])}, "
          f"stress {len(pack['stress_k_true'])}, "
          f"stability {len(pack['stab_q_k'])} realizations, "
          f"structural 4 configs")


def k_tau(subset, i, tau=0.5):
    """K_tau: number of slots whose derived existence p_i exceeds tau."""
    return int((EX[subset][i] > tau).sum())


def baseline_catalogue(i):
    """The model's own answer: MAP catalogue (first K_MAP slots)."""
    return pack["stress_maps"][i][:int(pack["stress_k_pred"][i])]


def evaluate(build_fn, idx=None, name="", quiet=False):
    """Score a decision layer build_fn(i) -> (n, 3) on stress signals idx."""
    idx = DEV if idx is None else idx
    cats = [build_fn(int(i)) for i in idx]
    truths = [TRUTHS["stress"][int(i)] for i in idx]
    sc = catalogue_scores(cats, truths)
    if not quiet:
        print(f"{name:<22} recall {sc['recall']:.3f}   "
              f"precision {sc['precision']:.3f}   F1 {sc['f1']:.3f}   "
              f"FP/mix {sc['fp_per_mixture']:.3f}   "
              f"exact-K {sc['exact_k_accuracy']:.3f}   "
              f"median|Δf| {fmt_mhz(sc['median_abs_df'])}")
    return sc


_LIVE_MODEL = None


def live_demo(seed=20260714, K=5, ckpt_dir=None):
    """Run the ACTUAL pretrained network on one fresh signal.

    Needs the released 464 MB checkpoint (see appendix A0 for the download
    commands); prints a friendly pointer if it is absent. The model is
    cached after the first call, so re-runs with new seeds are instant.
    Appendix A0 contains this same recipe written out step by step.
    """
    global _LIVE_MODEL
    import os
    import sys
    if ckpt_dir is None:   # repo layout first, then standalone-folder layout
        ckpt_dir = next((d for d in ["../pretrained_model/test_clariden",
                                     "pretrained_model/test_clariden"]
                         if os.path.exists(f"{d}/model_config.pt")),
                        "pretrained_model/test_clariden")
    needed = [f"{ckpt_dir}/model_config.pt", f"{ckpt_dir}/checkpoints"]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        print("(optional cell) pretrained checkpoint not found — no problem:")
        print("  everything in this session reads the precomputed pack.")
        print("  To run the network yourself, follow appendix A0 (two curl")
        print("  commands, 464 MB), then re-run this cell.")
        return
    import math
    import torch
    if _LIVE_MODEL is None:
        sys.path.insert(0, "..")
        from src.model import SlotFlow
        cfg = torch.load(f"{ckpt_dir}/model_config.pt", map_location="cpu",
                         weights_only=False)
        ckpt_file = sorted(f for f in os.listdir(f"{ckpt_dir}/checkpoints")
                           if f.endswith(".ckpt"))[-1]
        ckpt = torch.load(f"{ckpt_dir}/checkpoints/{ckpt_file}",
                          map_location="cpu", weights_only=False)
        state = {k.replace("model.", "").replace("_orig_mod.", ""): v
                 for k, v in ckpt["state_dict"].items()}
        model = SlotFlow(hidden_dim=cfg["hidden_dim"],
                         max_slots=cfg["max_slots"],
                         use_noise_encoder=cfg.get("use_noise_encoder", False))
        model.load_state_dict(state, strict=True)
        model.eval()
        _LIVE_MODEL = (model, cfg)
        print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
              f"parameters (cached for re-runs)")
    model, cfg = _LIVE_MODEL

    sys.path.insert(0, "..")
    from src.dataset import MultiSinusoidDataset
    ds = MultiSinusoidDataset(
        set_size=1, num_samples_long=cfg["num_samples_long"],
        tEnd_long=cfg["tEnd_long"], num_samples_short=cfg["num_samples_short"],
        tEnd_short=cfg["tEnd_short"], max_components=cfg["max_components"],
        freq_range=cfg["freq_range"], amp_range=cfg["amp_range"],
        min_freq_sep=0.01, noise_std=cfg["noise_std"], seed=seed,
        mode="inference", allowed_K_values=[K])
    x_long, x_short, k_true, comps, params, *_ = ds[0]
    import torch as _t
    with _t.no_grad():
        out = model(x_long[None], x_short[None])          # the forward pass
        q_live = _t.softmax(out["K_logits"], -1)[0]
        samp = model.flow.sample(200, context=out["context"])
    k_map = int(out["K_pred"][0])
    freqs_live = (samp[..., 3] / 3.0 + 2.75).median(dim=1).values
    print(f"fresh signal (seed {seed}):  K_true = {k_true}   "
          f"K_MAP = {k_map}   max q(K|x) = {q_live.max():.3f}")
    tru_f = sorted(float(f) for _, _, f in params[:k_true])
    est_f = sorted(float(f) for f in freqs_live)
    for tf, ef in zip(tru_f, est_f):
        print(f"  true f = {tf:.4f} Hz   slot posterior median = {ef:.4f} Hz"
              f"   (|Δf| = {fmt_mhz(abs(tf - ef))})")


def show_signal(subset, i, tau=0.5, ax=None, title=None, legend=False):
    """Spectrum + truth (dashed) + ACTIVE slots at threshold tau.

    Active = the K_tau prefix; matched slots blue ○, unmatched pink ◆.
    Slot markers sit in a strip at the top — their HEIGHT is arbitrary,
    only their frequency position carries information.
    """
    spec = pack[f"{subset}_spec"][i].astype(np.float32)
    kt = int(pack[f"{subset}_k_true"][i])
    truth = pack[f"{subset}_truth"][i][:kt]
    maps = pack[f"{subset}_maps"][i]
    ktau = k_tau(subset, i, tau)
    pairs, un_p, _ = match_catalogue(maps[:ktau], truth)
    matched = {s for s, _ in pairs}
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(FREQS_AXIS[BAND], spec[BAND], lw=0.8, color=C["ink"], alpha=0.55)
    top = ax.get_ylim()[1] * 1.12
    ax.set_ylim(None, top)
    for amp, phase, f in truth:
        ax.axvline(f, color=C["truth"], ls="--", lw=0.9, alpha=0.65)
        ax.plot(f, top * 0.97, "v", color=C["truth"], ms=8, mec="white",
                mew=0.5, zorder=6)
    for s in range(ktau):
        if s in matched:
            mark_pred(ax, maps[s, 2], top * 0.89, ms=8)
        else:
            ax.plot(maps[s, 2], top * 0.89, "D", color=C["alert"], ms=7,
                    zorder=6)
    ax.set_xlabel("frequency  [Hz]")
    kmap = int(pack[f"{subset}_k_pred"][i])
    title2(ax, title or f"{subset}[{i}]",
           f"K_true={kt}   K_MAP={kmap}   K_τ(τ={tau})={ktau}")
    if legend:
        ax.plot([], [], "v", color=C["truth"], label="true source")
        ax.plot([], [], "o", mfc="none", color=C["pred"], label="matched slot")
        ax.plot([], [], "D", color=C["alert"], label="unmatched slot")
        ax.legend(loc="lower right")
    return ax


def slot_table(subset, i, tau=0.5):
    """Text inspection of one signal. Active slots = K_tau prefix."""
    p = EX[subset][i]
    maps = pack[f"{subset}_maps"][i]
    stds = pack[f"{subset}_stds"][i]
    kt = int(pack[f"{subset}_k_true"][i])
    kmap = int(pack[f"{subset}_k_pred"][i])
    ktau = int((p > tau).sum())
    truth = pack[f"{subset}_truth"][i][:kt]
    pairs, un_p, un_t = match_catalogue(maps[:ktau], truth)
    match_of = dict(pairs)
    print(f"── {subset}[{i}]   K_true={kt}   K_MAP={kmap}   "
          f"K_τ(τ={tau})={ktau}")
    print(f"{'slot':>4} {'p_i':>6} {'f [Hz]':>9} {'σ_f [mHz]':>10} "
          f"{'A':>5}  status")
    for s in range(M):
        if s < ktau:
            if s in match_of:
                df = abs(maps[s, 2] - truth[match_of[s], 2])
                status = f"matched → source {match_of[s]} (|Δf|={fmt_mhz(df)})"
            else:
                status = "ACTIVE, UNMATCHED"
        else:
            status = "inactive"
        print(f"{s:>4} {p[s]:>6.2f} {maps[s, 2]:>9.4f} "
              f"{1e3 * stds[s, 2]:>10.2f} {maps[s, 0]:>5.2f}  {status}")
    if un_t:
        print("  ✗ unexplained true sources: " + ", ".join(
            f"f={truth[j, 2]:.4f} Hz (A={truth[j, 0]:.2f})" for j in un_t))
    print()
