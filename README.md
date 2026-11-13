# self_learning1 – self-learning neural-network solver for viscoplastic return mapping

Research code for a **self-learning neural-network (NN) solver** of the local return mapping of an
elasto-viscoplastic material with kinematic hardening at the integration points of a plate (plane stress).
Instead of a classical root finder (Newton / regula falsi), a recurrent network predicts the plastic multiplier
Δε_p and is **trained online on the physics residual** (consistency condition) at every solve – no labels, no
classical solution is used. The results are compared against the classical implicit solver.

> **TODO (author):** add the paper reference / citation and where the external data (weights, strain files) can be
> obtained – see [External data](#external-data).

## Contents

| File | Purpose |
|---|---|
| `Self_learning_Loeser_last_layer.ipynb` | **Main reproducible notebook.** Meta-pretrained hybrid spiking network (HSNN) frozen, only the last layer `Dense(1)` is trained online; forward pass + optimisation loop fused in one XLA-compiled call; compared with the classical solver. |
| `Self_learning_Loeser_SimpleRNN.ipynb` | Same solver with a small second-generation network (2 × SimpleRNN(32), random initialisation). Older settings – regenerate with `make_nb.py` to get the current ones. |
| `make_nb.py` | Generates the two notebooks above from the original notebooks (copies the material model and the classical solver verbatim). |
| `Self_learning_Loeser_plastiche_dehnung_berechnung.ipynb` | Original self-learning notebook (all network layers trained online; results of the paper run). |
| `Klassich_implicit.ipynb` | Classical implicit solver (regula falsi, `implicit_integration_vp_gnl_new`) – the reference. |
| `Hybrid_Spiking_Neuron.py`, `decoding.py` | Keras RNN cells `HSNN` (leaky integrate-and-fire, output spike·voltage) and `DSNN` (voltage read-out) with surrogate spike gradient. |
| `recurrent_LIF.py`, `second_gen_RNN.py`, `Klassi*RNN*.ipynb`, `Klassi*SNN*.ipynb` | Stand-alone network experiments, not used by the solver. |
| `Effort_towards_*.ipynb` | Earlier explicit-integration variants with isotropic damage. |

## Model in brief

- Material: E = 67400 MPa, ν = 0.32, yield stress 110 MPa, Norton-type viscosity (K = 3.42, N = 2.79,
  Δt = 1e-4), Armstrong–Frederick kinematic hardening (a = 3100, s = 110).
- Integration loop per time increment and integration point: elastic trial stress → yield check → plane-stress
  iteration (secant update of Δε₃₃ until σ₃₃ ≈ 0) → in every iteration one **local solve** for Δε_p →
  update of plastic/elastic strain, stress, back stress, equivalent plastic strain.
- NN local solve: Δε_p = max_val · sigmoid(w·φ + b), loss = (consistency residual)², Adam on (w, b) until
  loss < TOL. The weights carry over from solve to solve (self-learning); the frozen network provides φ and keeps
  its recurrent state per integration point.

## Setup

```bash
conda create -n self_learning python=3.10
conda activate self_learning
pip install -r requirements.txt
python -m ipykernel install --user --name self_learning --display-name "Python (self_learning)"
```

The notebooks run on the **CPU** by default (`USE_GPU = False`); for this small scalar optimisation loop the CPU
is about 8× faster than a GPU. For GPU use on Windows, TensorFlow 2.10 needs CUDA 11.2 and cuDNN 8.1
(e.g. `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1`).

## External data

The data is **not part of this repository**. Set the paths in the configuration cell (cell 2) of the generated
notebooks:

| Setting | Expected content |
|---|---|
| `WEIGHTS_PATH` | `meta_weights_HSNN_up_different_scaling_more_innerloop.pkl` – pickled list of 14 NumPy arrays: HSNN layers `[0:2]`, `[2:4]`, `[4:6]`, DSNN `[6:8]`, Dense(240) `[8:10]`, Dense(64) `[10:12]`, Dense(1) `[12:14]`. Not needed for the SimpleRNN notebook. |
| `STRAIN_DIR` | Folder `BVP4_100_els` with `strain1<l><m><n>.csv` for the 45 integration points of element 1 (l, m = 1…3 in-plane Gauss points, n = 1…5 section points). Each file: 30 rows × 6 columns (ε11, ε22, ε33, ε23, ε13, ε12), whitespace separated, **cumulative** total strain of the boundary value problem. The path is cyclic: loading (rows 0–9), load reversal from row 10. |

The original notebooks (`Self_learning_Loeser_plastiche_dehnung_berechnung.ipynb`, `Klassich_implicit.ipynb`,
`Effort_towards_*.ipynb`) contain **hard-coded absolute Windows paths**; adapt them before running (in the
original self-learning notebook: cell 1 training data, cell 3 weights, cell 9 strain files, cell 12 iteration
statistics).

## Reproducing the results

Run all cells of `Self_learning_Loeser_last_layer.ipynb` with the kernel "Python (self_learning)", or headless
from the repository root (the notebooks import `Hybrid_Spiking_Neuron.py` / `decoding.py` from there):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Self_learning_Loeser_last_layer.ipynb
```

Default configuration (cell 2): integration point (1, 1, 1) (`strain1111.csv`), `INCREMENTS = 30` with the actual
strain increments of the file (`STRAIN_PATH = 'file'`), `TOL = 1e-9` (residual² of the NN solve),
`LR = 0.003`, `USE_XLA = True`, `ADAM_RESET = 'never'`, flow direction `Z/J`. The notebook runs the integration
three times: with the NN solver, with the classical solver, and with the classical solver and the flow direction of
the original notebook (`Z/h`, for reference).

Expected summary (cells 13 and 15; values obtained on a Windows 11 workstation, CPU):

| Quantity | NN (self-learning, last layer) | Classical |
|---|---|---|
| Plastic increments | 9–10 (loading), 28–30 (reverse loading) | same |
| Local solves | 22, all converged | 22 |
| Adam steps per solve | 128 (mean), 266 (max) | – |
| Max. relative error of Δε_p vs. classical | 4.7·10⁻⁶ | – |
| Max. stress difference vs. classical | 2.5·10⁻⁵ MPa (σ up to 132 MPa) | – |
| Final equivalent plastic strain | 8.747571·10⁻⁴ | 8.747566·10⁻⁴ |
| Time per local solve (median) | ≈ 0.95 ms | ≈ 1.09 ms |

The first two NN solves include the one-time XLA compilation (≈ 2 s, one compile per `max_iter` value); they are
excluded from the timing. Timings depend on the machine; step counts and errors may differ in the last digits
with other hardware or library versions, the agreement with the classical solver should be of the same order.

## Regenerating / varying the notebooks

`make_nb.py` builds the notebooks; settings are passed as environment variables:

```bash
python make_nb.py Self_learning_Loeser_last_layer.ipynb hsnn          # pretrained HSNN + Dense(1)
python make_nb.py Self_learning_Loeser_SimpleRNN.ipynb simple_rnn     # 2 x SimpleRNN(32) + Dense(1)

# options (defaults in brackets)
GEN_INCREMENTS=30            # number of increments, at most 30 with GEN_STRAIN_PATH=file   [30]
GEN_STRAIN_PATH=file         # 'file': row t - row t-1;  'constant': row 0 every step (as in the original notebook)  [file]
GEN_TOL=1e-9                 # NN stop criterion residual² < TOL (original notebook: 1e-7)  [1e-9]
GEN_USE_XLA=True             # XLA compilation of the fused solve  [True]
GEN_ADAM_RESET=never         # 'solve' | 'increment' | 'never': when Adam's moments are reset  [never]
```

The paths, the integration points (`POINTS`) and the other settings can also be edited directly in cell 2 of a
generated notebook.
