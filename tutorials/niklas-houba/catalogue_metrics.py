"""Catalogue-level metrics and decision-layer helpers for Tutorial 2.

Works on plain numpy arrays from the predictions pack — no model needed.

Conventions
-----------
* q_k[j] = q(K = j+1 | x): the cardinality posterior over K in {1..M}.
* Slots are instantiated in index order, so slot i (0-based) is active
  iff K >= i+1. Hence the derived per-slot existence probability
      p_i = P(K >= i+1 | x) = sum_{j >= i} q_k[j],
  which is monotone non-increasing in i.
* A predicted catalogue is an array (n, 3) of [amp, phase, freq] rows.
* Matching between a predicted and a true catalogue is a Hungarian
  assignment on |f_pred - f_true|; a pair counts as a detection when
  |f_pred - f_true| <= tol_f. Frequency is the identity-carrying
  parameter for quasi-monochromatic sources.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

TOL_F = 0.005  # Hz; ~1.5x the Rayleigh limit 1/T = 1/300 Hz of the long stream


def existence_probs(q_k):
    """p_i = P(K >= i+1 | x) for each slot i. q_k: (..., M)."""
    return np.flip(np.cumsum(np.flip(q_k, axis=-1), axis=-1), axis=-1)


def k_from_threshold(q_k, tau):
    """K_hat(tau) = #{ i : p_i > tau }. tau=0.5 is the posterior-median K."""
    return (existence_probs(q_k) > tau).sum(axis=-1)


def match_catalogue(pred, truth, tol_f=TOL_F):
    """Match predicted rows to true rows on |delta f|.

    The assignment MAXIMIZES the number of within-tolerance detections
    (links with |delta f| > tol_f are penalized so heavily the solver only
    uses them when unavoidable), then minimizes |delta f| among the
    detections. A plain min-total-distance assignment would let a
    near-duplicate row corrupt its neighbours' pairings and undercount
    detections.

    Returns (pairs, unmatched_pred, unmatched_true) where pairs is a list
    of (i_pred, j_true) with |delta f| <= tol_f.
    """
    if len(pred) == 0 or len(truth) == 0:
        return [], list(range(len(pred))), list(range(len(truth)))
    cost = np.abs(pred[:, 2][:, None] - truth[:, 2][None, :])
    capped = np.where(cost <= tol_f, cost, 1e6)
    rows, cols = linear_sum_assignment(capped)
    pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] <= tol_f]
    got_p = {i for i, _ in pairs}
    got_t = {j for _, j in pairs}
    return (pairs,
            [i for i in range(len(pred)) if i not in got_p],
            [j for j in range(len(truth)) if j not in got_t])


def catalogue_scores(preds, truths, tol_f=TOL_F):
    """Aggregate scores over a list of (pred, truth) catalogue pairs.

    Returns dict with recall, precision, f1, false positives per mixture,
    exact-K accuracy, and median |delta f| / |delta A| on matched pairs.
    """
    n_det = n_true = n_pred = n_exact = 0
    fp = 0
    df, da = [], []
    for pred, truth in zip(preds, truths):
        pairs, un_p, _ = match_catalogue(pred, truth, tol_f)
        n_det += len(pairs)
        n_true += len(truth)
        n_pred += len(pred)
        fp += len(un_p)
        n_exact += int(len(pred) == len(truth))
        for i, j in pairs:
            df.append(abs(pred[i, 2] - truth[j, 2]))
            da.append(abs(pred[i, 0] - truth[j, 0]))
    recall = n_det / max(n_true, 1)
    precision = n_det / max(n_pred, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-12)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fp_per_mixture": fp / max(len(preds), 1),
        "exact_k_accuracy": n_exact / max(len(preds), 1),
        "median_abs_df": float(np.median(df)) if df else float("nan"),
        "median_abs_dA": float(np.median(da)) if da else float("nan"),
    }


def slot_distance(a, b, sigma_f=TOL_F, sigma_a=0.25):
    """Normalized source-space distance between two slot predictions,
    d = (df/sigma_f)^2 + (dA/sigma_a)^2. Raw units would let one
    parameter dominate — that is the point of the normalization."""
    return ((a[2] - b[2]) / sigma_f) ** 2 + ((a[0] - b[0]) / sigma_a) ** 2


def suppress_duplicates(cat, p, eps=4.0, sigma_f=TOL_F, sigma_a=0.25):
    """Catalogue-level non-maximum suppression: when two rows are closer
    than eps in normalized source space, keep the higher-existence one.
    cat: (n, 3); p: (n,) existence probabilities. Returns kept indices."""
    order = np.argsort(-np.asarray(p))
    kept = []
    for i in order:
        if all(slot_distance(cat[i], cat[j], sigma_f, sigma_a) >= eps
               for j in kept):
            kept.append(i)
    return sorted(kept)


def build_catalogue(maps, q_k, tau=0.5, nms_eps=None,
                    sigma_f=TOL_F, sigma_a=0.25):
    """The decision layer participants tune in the challenge.

    maps: (M, 3) per-slot MAP parameters; q_k: (M,) cardinality posterior.
    Threshold the derived existence probabilities at tau, then optionally
    apply duplicate suppression. Returns the catalogue (n, 3).
    """
    p = existence_probs(q_k)
    active = np.where(p > tau)[0]
    cat, pa = maps[active], p[active]
    if nms_eps is not None and len(cat) > 1:
        keep = suppress_duplicates(cat, pa, nms_eps, sigma_f, sigma_a)
        cat = cat[keep]
    return cat
