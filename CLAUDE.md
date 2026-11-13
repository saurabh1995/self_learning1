# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code (Jupyter notebooks, German comments/names) for a **self-learning neural-network solver of the
viscoplastic return mapping** at the integration points of a plate (plane stress). Material: elasto-viscoplastic
with Norton-type rate law (`stress_yield=110`, `stress_K=3.42`, `N=2.79`, `timeincrement=1e-4`) and
Armstrong-Frederick kinematic hardening (`a_unit=3100`, `s_unit=110`), `E=67400`, `ν=0.32`.
Instead of Newton/regula falsi, a (meta-)pretrained recurrent network predicts the plastic multiplier Δε_p and is
trained **online** on the consistency residual at every solve (no labels). There is no build, lint or test suite;
correctness is checked by comparing against the classical solver inside the notebooks.

## Environment and commands

The base Anaconda env cannot run TensorFlow (protobuf 6 conflict), `tf_gpu` has numpy 2 (TF 2.10 crashes).
Use the conda env **`self_learning`** (clone of `tf_gpu` + `numpy<2`, `tensorflow-addons==0.19.0`,
`keras-lmu==0.5.0`, scikit-learn, scipy, sympy, pandas, matplotlib), Jupyter kernel **"Python (self_learning)"**.

```bash
PY=/c/Users/Tandale/anaconda3/envs/self_learning/python.exe
# execute a notebook (writes outputs into it) -- run from the repo root, the notebooks import decoding.py / Hybrid_Spiking_Neuron.py
$PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Self_learning_Loeser_last_layer.ipynb
# regenerate the last-layer notebooks (see "Generated notebooks")
GEN_USE_XLA=True GEN_ADAM_RESET=never $PY make_nb.py Self_learning_Loeser_last_layer.ipynb hsnn
$PY make_nb.py Self_learning_Loeser_SimpleRNN.ipynb simple_rnn
```

To execute a notebook copy outside the repo, set `PYTHONPATH` to the repo root. When exec'ing notebook cells from a
script, register the cell source in `linecache` first, otherwise AutoGraph cannot convert the `tf.function`s
(`OperatorNotAllowedInGraphError`).

## External data (not in the repo, absolute paths hard-coded in the notebooks)

- Pretrained weights: `...\PostDoc\Writing\Meta_Learning\Trained_Model\meta_pretrainiertes_model\meta_weights_HSNN_up_different_scaling_more_innerloop.pkl`
  (14 arrays: `[0:2],[2:4],[4:6]` HSNN layers, `[6:8]` DSNN, `[8:10]` Dense 240, `[10:12]` Dense 64, `[12:14]` Dense 1).
- Strains per integration point: `...\Phd\Writing\SLMU_paper\BVPs\Stress-Strain_Data_csv\BVP4_100_els\strain1<l><m><n>.csv`
  (whitespace separated, 30 rows of **cumulative** total strain from the BVP, cyclic: loading rows 0–9, load reversal
  from row 10). The original notebook applies row 0 as a constant increment every step (monotonic loading); the
  generated notebooks use the actual increments `row t − row t−1` (`STRAIN_PATH='file'`, default).
- Training data for the unused scaler (cell 1 of the original notebook): `...\Meta_Learning\Training_data+MAML\input<k>.csv/output<k>.csv`.

## Architecture

**Main notebook** `Self_learning_Loeser_plastiche_dehnung_berechnung.ipynb` (paper results, keep outputs; do not
execute it in place). Cells: 0 imports, 1 unused training-data/scaler load, 3 network build from the `.pkl`,
5 material parameters + helpers, 7 local NN solve `train_LSTM_static_KI_` (`train_LSTM_static_KI` without underscore
is dead/broken), 9 integration loop.

- **State storage:** everything lives in the flat dict `mat_params`, keyed by quantity name + the string
  `str(i+1)+str(l+1)+str(m+1)+str(n+1)` (element `i`, in-plane Gauss point `l,m` 3×3, section point `n` 1..5 →
  45 points). Recurrent network states per point are also stored there (`state_h_…`, `state_c_…`). `i` is not set
  by the loop cell; it leaks from the `for i in range(1)` in the material cell.
- **Integration loop (per increment t, per point):** elastic trial stress → yield check → plane-stress iteration
  `p` (secant update of Δε₃₃ until σ₃₃≈0, tol 1e-5) → in every `p` one **local solve** for Δε_p → update of
  plastic/elastic strain, stress, back stress (`X₃₃=0`), EQPLAS. Recurrent states are stored once per increment.
- **Local NN solve:** network output → `scaled_sigmoid(·, 0, max_val)` (upper bound from `Delta_pl_U1/U2`), loss =
  squared consistency residual `J_N/α_N − h_N`; Adam(0.003) re-created every `p`, stop at loss < 1e-7.
- **Networks:** `Hybrid_Spiking_Neuron.HSNN` (LIF cell, output spike·voltage) ×3 + `decoding.DSNN` (voltage readout)
  + Dense(240, tanh) + Dense(64, tanh) + Dense(1); spikes use a Heaviside with atan surrogate gradient
  (`SpikeFunction`). `recurrent_LIF.py` (RSNN), `second_gen_RNN.py` and the `Klassi*RNN/SNN` notebooks are
  standalone experiments not used by the solver.
- **Flow direction:** the original notebook uses `N_plas = Z_N/h_N`; the consistent normal is `3/2·Z_N/J_N`
  (`J_N/α_N = h_N` at consistency), so the original over-applies plastic flow by α_N (1.03–1.15, ~11 % lower EQPLAS
  after 30 increments). `Klassich_implicit.ipynb` (classical regula-falsi solver `implicit_integration_vp_gnl_new`,
  cell 4) and the generated notebooks use `Z_N/J_N`.

**Generated notebooks** (branch `last-layer-self-learning`): `Self_learning_Loeser_last_layer.ipynb` (HSNN) and
`Self_learning_Loeser_SimpleRNN.ipynb` (2×SimpleRNN(32), random init) are produced by **`make_nb.py`**, which copies
the material cell (original cell 5) and the classical solver (Klassich cell 4) verbatim. Change the generator, not
the generated notebooks, when the change should apply to both.
- Frozen network + trainable `Dense(1)` (`W`, `B` float64 variables, never reset during a run = self-learning).
- `fused_solve`: one `tf.function(jit_compile=USE_XLA)` call per solve = forward pass + Adam loop
  (`for k in tf.range(max_iter)` with `break`). Inputs packed into one float64 vector (`pack_params`).
  Adam moments persist in variables; `ADAM_RESET` = `'solve'` | `'increment'` | `'never'`.
- `run(solver, flow)` drives the full integration with `'nn'` or `'classical'` local solves; both share the same
  strain update, so the comparison isolates the local solver. Summary cell reports per-solve timing and Δε_p/σ errors.
- Performance facts: run on CPU (`USE_GPU=False`, GPU is ~8× slower for this scalar loop); without XLA each Adam
  step costs ~110 µs, with XLA the ~100 steps are nearly free and a solve is ~0.9 ms (classical pure-Python solver
  ~1.0 ms). XLA compiles once per distinct `max_iter` value (5000 for `p==0`, 1000 otherwise, ~2 s total), so
  exclude those calls when timing.

`Effort_towards_*.ipynb`: earlier explicit-integration variants (with isotropic damage, own RSNN/DSNN classes,
untrained network, reads `strain44111.csv` from another path).

`.ipynb_checkpoints/` and `__pycache__/` are git-ignored (they were tracked until the `.gitignore` was added; older
commits still contain them).
