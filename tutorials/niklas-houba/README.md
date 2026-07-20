# SlotFlow tutorials — ESA "Machine Learning Methods for LISA"

Self-contained folder: everything both tutorials need is inside, with
relative paths. No downloads required (one optional exception, below).

## Quick start

```bash
pip install torch numpy scipy matplotlib scikit-learn jupyter ipywidgets
# (tested with torch 2.5.1, numpy 2.1.3, scipy 1.14.1, matplotlib 3.10.1,
#  scikit-learn 1.6.1 — nearby versions should be fine)

jupyter lab          # start FROM THIS FOLDER, then open the notebooks
```

| notebook | what it is | runtime |
|---|---|---|
| `slotflow_tutorial_1_toy.ipynb` | Session 1 (40 min): train a toy set-prediction model, break it with an ordered loss, fix it with Hungarian matching. One TODO. | all cells < 10 s; whole notebook ~1 min |
| `slotflow_tutorial_2_diagnose.ipynb` | Session 2 (60 min): diagnose the pretrained SlotFlow from a precomputed prediction pack — slot tables, failure gallery, catalogue challenge. One coding task. | everything < 2 s per cell |
| `slotflow_tutorial_*_solution.ipynb` | The same notebooks with reference implementations, fully executed (all outputs and figures included — readable without running anything). | — |

Both run CPU-only; no GPU needed. The `*_toy`/`*_diagnose` versions are
what workshop participants work through (they contain the TODOs); the
`*_solution` versions are the reveals.

## What's in here

- `toy/` — the Tutorial 1 simulator, model, and pre-generated data +
  staged checkpoints (so every training cell is skippable).
- `predictions_pack.npz` (21 MB) + `gallery.json` — the pretrained
  model's offline outputs on 2,000 test signals; Tutorial 2 reads only
  these. `viz.py`, `t2_helpers.py`, `catalogue_metrics.py` — shared
  plotting/metrics helpers.
- `src/` + `pretrained_model/test_clariden/model_config.pt` — only used
  by the **optional** "run the model yourself" cells (Tutorial 2, §1 and
  appendix A0).

## Optional: run the pretrained network live

Everything works without this. If you want the §1/A0 cells to run the
actual network (instead of printing a pointer), fetch the released
weights (464 MB) into this folder — also needs `pip install nflows`:

```bash
curl -L --create-dirs -o pretrained_model/test_clariden/checkpoints/best_model.ckpt \
  https://github.com/nhouba/slotflow-inference/releases/download/v1.0.0/best_model.ckpt
```

## Notes

- Notebooks expect the working directory to be this folder (starting
  Jupyter here is enough; the setup cell also copes with the parent).
- Tutorial 2's setup cell only tries to download the pack if
  `predictions_pack.npz` is missing — it isn't, so no network access is
  needed.
- Questions: Niklas Houba <nhouba@phys.ethz.ch>. Paper: arXiv:2511.23228.
