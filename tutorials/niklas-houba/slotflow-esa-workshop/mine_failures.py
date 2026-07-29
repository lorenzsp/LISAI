"""Mine the predictions pack for the Tutorial 2 gallery and exemplars.

Failure labels are NOT mutually exclusive — one signal often exhibits
several at once (a merge implies an undercount, which usually implies a
miss). The gallery therefore uses MULTI-LABEL diagnosis: participants mark
every applicable label, and the reveal shows the checklist matrix.

Selection criteria are in the open (this file ships with the repo) and
base rates are stored, so the gallery cannot read as cherry-picked.

    python workshop/mine_failures.py        -> workshop/gallery.json

Also mined here: the NMS help/hurt exemplars and the spectral-rescue
success/false-positive exemplars used later in the notebook.
"""

import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalogue_metrics import TOL_F, match_catalogue, slot_distance  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(ROOT, "predictions_pack.npz")

LABELS = ["miss", "merge", "duplicate", "undercount", "overcount",
          "exactK_wrong"]


def label_signal(pred, truth, k_pred, k_true, tol=TOL_F):
    """Multi-label diagnosis of one signal's MAP catalogue."""
    pairs, un_p, un_t = match_catalogue(pred, truth, tol)
    labels = set()
    if un_t:
        labels.add("miss")
    if k_pred < k_true:
        labels.add("undercount")
    if k_pred > k_true:
        labels.add("overcount")
    if k_pred == k_true and un_p and un_t:
        labels.add("exactK_wrong")
    # duplicate = a REDUNDANT claim: an unmatched prediction sitting within
    # tol of a true source that some other prediction already explains.
    # (Counting predictions near each truth without consulting the assignment
    # fires on well-separated pairs whose members are merely closer together
    # than tol — a metric artifact, not a model failure.)
    matched_t = {j for _, j in pairs}
    for i in un_p:
        if any(abs(pred[i, 2] - truth[j, 2]) <= tol for j in matched_t):
            labels.add("duplicate")
    for j1 in range(len(truth)):
        for j2 in range(j1 + 1, len(truth)):
            f1, f2 = truth[j1, 2], truth[j2, 2]
            if abs(f1 - f2) <= 2 * tol:
                lo, hi = min(f1, f2) - tol, max(f1, f2) + tol
                if sum(lo <= pred[i, 2] <= hi for i in range(len(pred))) == 1:
                    labels.add("merge")
    return labels, pairs, un_p, un_t


def entry(subset, i, labels, kt, kp, pairs, un_p, un_t):
    return {"subset": subset, "index": int(i), "labels": sorted(labels),
            "k_true": kt, "k_pred": kp, "n_matched": len(pairs),
            "unmatched_slots": [int(x) for x in un_p],
            "unmatched_sources": [int(x) for x in un_t]}


def main():
    pack = np.load(PACK)
    rates, per_signal = {}, {"stress": []}
    for subset in ["nominal", "stress"]:
        k_true = pack[f"{subset}_k_true"]
        k_pred = pack[f"{subset}_k_pred"]
        maps = pack[f"{subset}_maps"]
        truth = pack[f"{subset}_truth"]
        n = len(k_true)
        counts = {c: 0 for c in LABELS}
        for i in range(n):
            kt, kp = int(k_true[i]), int(k_pred[i])
            labels, pairs, un_p, un_t = label_signal(
                maps[i, :kp], truth[i, :kt], kp, kt)
            for c in labels:
                counts[c] += 1
            if subset == "stress":
                per_signal["stress"].append(
                    entry(subset, i, labels, kt, kp, pairs, un_p, un_t))
        rates[subset] = {c: {"count": v, "rate": v / n}
                         for c, v in counts.items()}
        rates[subset]["k_error"] = {
            "count": int((k_true != k_pred).sum()),
            "rate": float((k_true != k_pred).mean())}
        print(f"[{subset}] n={n}  K-error {rates[subset]['k_error']['rate']:.2%}  "
              + "  ".join(f"{c} {v/n:.1%}" for c, v in counts.items()))

    # ---- mystery selection: 4 DISTINCT signals, jointly covering as many
    # labels as possible; one deliberately compound (>=3 labels); prefer
    # readable K (<= 8 sources)
    cands = [e for e in per_signal["stress"] if e["labels"]]
    compound = sorted((e for e in cands if len(e["labels"]) >= 3),
                      key=lambda e: (e["k_true"] > 8, -len(e["labels"])))
    mystery, covered, used = [], set(), set()
    if compound:
        mystery.append(compound[0])
        covered |= set(compound[0]["labels"])
        used.add(compound[0]["index"])
    # the duplicate is the rarest failure (~1%) and the one NMS addresses in
    # the challenge, so make sure ONE case is duplicate-led rather than letting
    # it appear only inside the compound pile-up: prefer the fewest extra
    # labels and a readable K.
    dup_led = sorted((e for e in cands
                      if "duplicate" in e["labels"] and e["index"] not in used),
                     key=lambda e: (e["k_true"] > 8, len(e["labels"])))
    if dup_led:
        mystery.append(dup_led[0])
        covered |= set(dup_led[0]["labels"])
        used.add(dup_led[0]["index"])
    seen_sets = {tuple(sorted(m["labels"])) for m in mystery}
    while len(mystery) < 4:
        best = max((e for e in cands if e["index"] not in used),
                   key=lambda e: (len(set(e["labels"]) - covered),
                                  tuple(sorted(e["labels"])) not in seen_sets,
                                  e["k_true"] <= 8, -len(e["labels"])),
                   default=None)
        if best is None:
            break
        mystery.append(best)
        covered |= set(best["labels"])
        seen_sets.add(tuple(sorted(best["labels"])))
        used.add(best["index"])
    order = np.random.default_rng(11).permutation(len(mystery))
    mystery = [mystery[j] for j in order]
    print("mystery:", [(e["index"], e["labels"]) for e in mystery])

    # ---- NMS exemplars: one true duplicate (NMS helps), one genuine close
    # pair matched by two slots (NMS would delete a real detection).
    # Exclude the mystery indices and the fixed slot-table demo indices so
    # no example appears twice in the notebook.
    excluded = used | {0, 1, 4}
    helps = hurts = None
    for e in per_signal["stress"]:
        if e["index"] in excluded:
            continue
        if helps is None and "duplicate" in e["labels"]:
            helps = {"subset": "stress", "index": e["index"]}
        if hurts is None and "duplicate" not in e["labels"]:
            i, kt, kp = e["index"], e["k_true"], e["k_pred"]
            maps = pack["stress_maps"][i][:kp]
            tru = pack["stress_truth"][i][:kt]
            pairs, _, _ = match_catalogue(maps, tru)
            match_of = dict(pairs)
            for s1 in match_of:
                for s2 in match_of:
                    if s1 < s2 and match_of[s1] != match_of[s2] and \
                            slot_distance(maps[s1], maps[s2]) < 4.0:
                        hurts = {"subset": "stress", "index": i,
                                 "slots": [int(s1), int(s2)]}
        if helps and hurts:
            break

    # ---- spectral-rescue exemplars.
    # success: at the working threshold (0.35) an added peak recovers a
    #   genuinely missed source.
    # false positive: rescue's failure mode. At 0.35 no added peak lands
    #   >tol from every true source (measured) — its real FP mode is
    #   double-counting an already-detected source; and at a permissive
    #   threshold noise peaks enter. Mine both, record which was found.
    freqs_axis = np.fft.rfftfreq(3000, d=0.1)
    band = (freqs_axis > 2.45) & (freqs_axis < 3.05)
    fb = freqs_axis[band]

    def added_peaks(i, hf):
        s = pack["stress_spec"][i][band].astype(np.float32)
        kp = int(pack["stress_k_pred"][i])
        base = pack["stress_maps"][i][:kp]
        pk, _ = find_peaks(s, height=hf * s.max(), distance=2)
        return [fb[j] for j in pk
                if np.abs(base[:, 2] - fb[j]).min() > TOL_F]

    success = false_pos = None
    for e in per_signal["stress"]:
        i, kt = e["index"], e["k_true"]
        if i in excluded or "miss" not in e["labels"]:
            continue
        tru = pack["stress_truth"][i][:kt]
        for f_add in added_peaks(i, 0.35):
            if np.abs(tru[:, 2] - f_add).min() <= TOL_F:
                success = {"subset": "stress", "index": i,
                           "f_added": float(f_add), "height_frac": 0.35}
                break
        if success:
            break
    for e in per_signal["stress"]:
        i, kt = e["index"], e["k_true"]
        if i in excluded or (success and i == success["index"]):
            continue
        tru = pack["stress_truth"][i][:kt]
        for hf in (0.35, 0.15):
            for f_add in added_peaks(i, hf):
                if np.abs(tru[:, 2] - f_add).min() > TOL_F:
                    false_pos = {"subset": "stress", "index": i,
                                 "f_added": float(f_add), "height_frac": hf}
                    break
            if false_pos:
                break
        if false_pos:
            break

    out = {"tol_f": TOL_F, "labels": LABELS, "rates": rates,
           "mystery": mystery,
           "nms_cases": {"helps": helps, "hurts": hurts},
           "rescue_cases": {"success": success, "false_positive": false_pos}}
    out_path = os.path.join(ROOT, "gallery.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"nms helps={helps} hurts={hurts}")
    print(f"rescue success={success} false={false_pos}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
