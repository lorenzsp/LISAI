# SlotFlow: From Toy Problems to LISA

Hands-on tutorial materials from the **LISA AI Hackathon: Machine Learning
Methods for LISA** (ESA ESTEC) — a 20-minute introduction and two
tutorials on **amortized trans-dimensional inference**: jointly inferring
*how many* sources are in a signal and *what* they are.

> train a toy set-prediction model → understand permutation-invariant
> matching → introduce unknown K → diagnose a real pretrained model

Everything runs **CPU-only**, locally or on Google Colab, and the repo is
self-contained — clone it and you have all data included.

## Run it on Google Colab — no install, nothing to download

Click a badge, then run the first cell. It clones this repository (data
included) into the Colab session and everything else just runs; no GPU and
no LISA software stack are needed.

| tutorial | start here | reference solution |
|---|---|---|
| **Tutorial 1** (40 min) — build the mechanism: break an ordered loss on purpose, then fix it with Hungarian matching *(one coding TODO)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhouba/slotflow-esa-workshop/blob/main/slotflow_tutorial_1_toy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhouba/slotflow-esa-workshop/blob/main/slotflow_tutorial_1_solution.ipynb) |
| **Tutorial 2** (60 min) — diagnose the pretrained SlotFlow: failure gallery, calibration, and the Catalogue Challenge *(one coding task)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhouba/slotflow-esa-workshop/blob/main/slotflow_tutorial_2_diagnose.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhouba/slotflow-esa-workshop/blob/main/slotflow_tutorial_2_solution.ipynb) |

The solution notebooks are already executed, so they are also readable
without running anything.

Colab sessions are ephemeral: your edits live in *your* Colab copy
(`File → Save a copy in Drive`), not in this repository.

## Run it locally

```bash
git clone https://github.com/nhouba/slotflow-esa-workshop.git
cd slotflow-esa-workshop
pip install torch numpy scipy matplotlib scikit-learn jupyter ipywidgets
jupyter lab
```

## The materials

| file | what it is |
|---|---|
| `slides/Introduction-SlotFlow.pdf` | the 20-minute introduction |
| `slotflow_tutorial_1_toy.ipynb` | **Tutorial 1 (40 min):** train a tiny slot-based set predictor on toy spectral sources; break it with an ordered loss, watch it hedge to the analytic floor, then fix it by implementing Hungarian matching (the session's one coding TODO). |
| `slotflow_tutorial_2_diagnose.ipynb` | **Tutorial 2 (60 min):** diagnose the pretrained SlotFlow model from a precomputed prediction pack — slot tables (K_true vs K_MAP), slot-identity experiments, a multi-label failure gallery, and the Catalogue Challenge: improve the decision layer and beat the model's own catalogue. |
| `slotflow_tutorial_*_solution.ipynb` | the same notebooks with reference implementations, fully executed — readable without running anything |

Both tutorials follow the same discipline: *representation ≠ inference ≠
matching ≠ decision ≠ evaluation* — failures are engineered on purpose so
you practice naming which layer failed.

## What's under the hood

- `toy/` — the Tutorial 1 simulator, model, and pre-generated data with
  staged checkpoints (every training cell is skippable).
- `predictions_pack.npz` + `gallery.json` — the pretrained model's
  offline outputs on 2,000 test signals (1,200 in-distribution, 800
  stressed) plus controlled stability experiments; Tutorial 2 reads only
  these.
- `viz.py`, `t2_helpers.py`, `catalogue_metrics.py` — shared plotting and
  catalogue-scoring helpers (tolerance-aware detection matching,
  existence probabilities derived from q(K|x), duplicate suppression).
- `nb_t1.py`, `nb_t2.py`, `build_notebooks.py` — the notebooks are
  generated from these (edit these, not the `.ipynb` files):
  `python build_notebooks.py`.

## Optional: run the pretrained network live

The tutorials never require it, but Tutorial 2's §1 and appendix A0 can
run the actual model on fresh signals if you fetch the released weights
(464 MB; also `pip install nflows`):

```bash
curl -L --create-dirs -o pretrained_model/test_clariden/checkpoints/best_model.ckpt \
  https://github.com/nhouba/slotflow-inference/releases/download/v1.0.0/best_model.ckpt
```

With the checkpoint in place you can also regenerate the prediction pack
from scratch (`python make_predictions_pack.py`, ~12 min on a laptop CPU)
and re-mine the failure gallery (`python mine_failures.py`).

## The model

SlotFlow (slot-based conditional normalizing flows for amortized
trans-dimensional inference):

```bibtex
@misc{houba2025slotflowamortizedtransdimensionalinference,
      title={SlotFlow: Amortized Trans-Dimensional Inference with Slot-Based Normalizing Flows},
      author={Niklas Houba and Giovanni Giarda and Lorenzo Speri},
      year={2025},
      eprint={2511.23228},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2511.23228},
}
```

Main research code: [github.com/nhouba/slotflow-inference](https://github.com/nhouba/slotflow-inference)

## Contact

Niklas Houba — nhouba@phys.ethz.ch

## License

MIT — see [LICENSE](LICENSE).
