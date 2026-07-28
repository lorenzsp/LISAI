# Neural Posterior Estimation for LISA

Tutorial material for the **AI for LISA Hackathon**, ESA/ESTEC, Noordwijk — Tuesday morning,
Stephen Green. Build a normalizing flow from scratch on a simplified LISA galactic binary,
validate it, extend it to a prior twenty times wider by heterodyning, then do the same thing with
DINGO in four commands.

## Quick start (Colab)

Open a notebook and **Runtime → Run all**:

| | | |
|---|---|---|
| Part 1 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stephengreen/LISAAI-Hackathon-ESTEC/blob/main/part1_npe.ipynb) | NPE from scratch, then validate it |
| Part 2a | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stephengreen/LISAAI-Hackathon-ESTEC/blob/main/part2a_conditioning.ipynb) | a wider prior, and how to survive it |
| Part 2b | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stephengreen/LISAAI-Hackathon-ESTEC/blob/main/part2b_dingo.ipynb) | the same method as DINGO, stored outputs included |

Select **Runtime → Change runtime type → T4 GPU**. The networks in Parts 1 and 2a train live,
about five and three minutes on a T4; set `TRAIN_LIVE = False` to load a stored network from
`checkpoints/` instead.

## Running on a laptop

With [uv](https://docs.astral.sh/uv/) installed:

```
git clone https://github.com/stephengreen/LISAAI-Hackathon-ESTEC.git
cd LISAAI-Hackathon-ESTEC
uv run jupyter lab
```

`uv run` builds the pinned environment from `uv.lock` on first use. Without uv: make a fresh
virtual environment (Python ≥ 3.10), `pip install glasflow corner torch matplotlib scipy jupyter`,
and open the notebooks. Training on a laptop CPU takes about seventeen minutes (Part 1) and ten
(Part 2a); `TRAIN_LIVE = False` skips it.

Part 2b needs DINGO in an environment of its own — `pip install dingo-gw jupyter`, which pulls
LALSuite, bilby and gwpy — and its first cell does that install automatically on Colab. Its
pipeline cells run anywhere; the inference sections sample from a trained network that is not
distributed with the repository, so their outputs ship stored in the notebook.

## Contents

| | |
|---|---|
| `presentation-LISAAI-ESTEC.pdf` | the morning's talk |
| `part1_npe.ipynb` | Part 1 — NPE from scratch on a galactic binary; validated by MCMC, importance sampling, and a P–P test. Exercises unsolved |
| `part2a_conditioning.ipynb` | Part 2a — heterodyning and a tile scan cover the widened prior with the frozen Part 1 network. Exercises unsolved |
| `part2b_dingo.ipynb` | Part 2b — the DINGO pipeline end to end, on an injection and on GW150914 |
| `solutions/` | Parts 1 and 2a with the exercises worked and all outputs stored |
| `checkpoints/` | trained networks, loaded when `TRAIN_LIVE = False` |
| `dingo/` | the DINGO settings files Part 2b drives |
| `results/` | stored `dingo_pipe` output for GW150914, and the public [GWTC-1 samples](https://dcc.ligo.org/LIGO-P1800370/public) |
| `gb_simulator.py`, `gb_wide.py` | the simulator as importable modules, with self-tests; the notebooks define everything inline |

Thursday's session (Max Dax) solves the same galactic-binary problem with flow matching in
`dingo.core`.
