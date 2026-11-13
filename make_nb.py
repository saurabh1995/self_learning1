"""Generate the last-layer self-learning notebooks from the original notebooks.

Usage (from any directory):
    python make_nb.py <output.ipynb> [hsnn|simple_rnn]

    hsnn       -> e.g. Self_learning_Loeser_last_layer.ipynb  (meta-pretrained HSNN + Dense(1))
    simple_rnn -> e.g. Self_learning_Loeser_SimpleRNN.ipynb   (2 x SimpleRNN(32), random init, + Dense(1))

Defaults written into the configuration cell (optional environment variables):
    GEN_USE_XLA    = True | False                   (default True)
    GEN_ADAM_RESET = 'solve' | 'increment' | 'never'  (default 'never')
    GEN_INCREMENTS = number of time increments          (default 30, at most 30 with 'file')
    GEN_TOL        = NN stop criterion residual**2 < TOL (default 1e-9; original notebook 1e-7)
    GEN_ELEMENT    = BVP element whose strain<element><l><m><n>.csv files are used (default 1)
    GEN_STRAIN_PATH = 'file' | 'constant'               (default 'file': actual increments row t - row t-1;
                                                         'constant': row 0 every step as in the original notebook)

Copied verbatim (checked by asserts):
    - material-model cell 5 of Self_learning_Loeser_plastiche_dehnung_berechnung.ipynb
    - classical solver cell 4 of Klassich_implicit.ipynb (implicit_integration_vp_gnl_new)

The generated notebook is not executed; run it in Jupyter (kernel "Python (self_learning)") or with
    jupyter nbconvert --to notebook --execute --inplace <output.ipynb>
"""
import json, os, sys
REPO = os.path.dirname(os.path.abspath(__file__))   # the repository root (this file lives there)
orig = json.load(open(os.path.join(REPO, "Self_learning_Loeser_plastiche_dehnung_berechnung.ipynb"), encoding="utf-8"))
klass = json.load(open(os.path.join(REPO, "Klassich_implicit.ipynb"), encoding="utf-8"))
src = lambda nb, k: ''.join(nb['cells'][k]['source'])

material_cell = src(orig, 5)
assert 'def gen_input' in material_cell and "mat_params['stress_K'" in material_cell
classical_cell = src(klass, 4)
assert classical_cell.lstrip().startswith('def implicit_integration_vp_gnl_new')

ARCH = sys.argv[2] if len(sys.argv) > 2 else 'hsnn'     # 'hsnn' | 'simple_rnn'
assert ARCH in ('hsnn', 'simple_rnn')

cells = []
md = lambda s: cells.append(dict(cell_type='markdown', metadata={}, source=s))
code = lambda s: cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s))

if ARCH == 'simple_rnn':
    md("""# Self-learning solver with a small second-generation RNN (2 x SimpleRNN(32) + Dense(1))

Variant of `Self_learning_Loeser_last_layer.ipynb` (same loop, residual, flow direction Z/J and classical comparison);
only the network is different:

* **2 x `SimpleRNN(32)`** (tanh), **randomly initialised** (fixed seed), **no pretraining**, kept **frozen**:
  together with its recurrent state per Gauss point it is a fixed random encoding (echo-state / random-feature
  network) of the trial deviatoric stress and back stress, evaluated **once per solve** -> 32-dim feature vector.
* Only the output layer **`Dense(1)`** (32 weights + 1 bias, random init) is trained **online with Adam**,
  Δε_p = max_val · sigmoid(w·φ + b), inside **one compiled `tf.function` graph** per solve.
* The last-layer weights are **not reset** between solves / Gauss points (self-learning).
* Flow direction `N_plas = Z_N / J_N` (as in `Klassich_implicit.ipynb`); the classical solver is evaluated on the
  same trial states and drives a reference run.
""")
else:
  md("""# Self-learning solver: only the last layer is trained (tf.function graph)

Variant of `Self_learning_Loeser_plastiche_dehnung_berechnung.ipynb`.

* The meta-pretrained hybrid spiking network (3x `HSNN` + `DSNN` + `Dense(240)` + `Dense(64)`) is **frozen**.
  For one return-mapping solve its input and recurrent states are fixed, so it is evaluated **once per solve** and
  yields a 64-dim feature vector.
* Only the output layer `Dense(1)` (64 weights + 1 bias) is optimised online. Δε_p = max_val · sigmoid(w·φ + b).
* Frozen forward pass **and** the whole optimisation loop (residual, exact gradient, Adam update, convergence check)
  run as **one fused, XLA-compiled `tf.function` call** per solve (`fused_solve`); all float inputs are packed into
  one vector. Adam's state persists between solves (`ADAM_RESET`, default `'never'`).
* The consistency residual is written entirely in TF ops (float64), so the gradient through J_N is exact.
* The last-layer weights are **not reset** between solves / Gauss points (self-learning, as in the original).
* At every solve the author's classical regula-falsi solver (`Klassich_implicit.ipynb`) is evaluated on the
  same trial state for comparison; a second run drives the whole integration with the classical solver.
* **Flow direction:** `N_plas = Z_N / J_N` as in `Klassich_implicit.ipynb`. The original self-learning notebook uses
  `Z_N / h_N`, which is larger by the factor alpha_N (consistency: J_N / alpha_N = h_N, normal n = 3/2 · Z_N / J_N).
  A third run (classical solver, `Z_N / h_N`) shows the effect of the original formulation.
""")

code("""# 0. Libraries
import os, copy, time, pickle
import numpy as np
import tensorflow as tf
USE_GPU = False   # tiny scalar graph loop: CPU is ~8x faster than GPU (kernel-launch overhead)
if not USE_GPU:
    tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from decoding import DSNN
from Hybrid_Spiking_Neuron import HSNN
%matplotlib inline""")

net_config = (r"""SEED         = 0         # random initialisation of the SimpleRNN layers and of Dense(1)
UNITS        = 32        # units per SimpleRNN layer
""" if ARCH == 'simple_rnn' else
r"""WEIGHTS_PATH = r"C:\Users\Tandale\sciebo\olddata\PostDoc\Writing\Meta_Learning\Trained_Model\meta_pretrainiertes_model\meta_weights_HSNN_up_different_scaling_more_innerloop.pkl"
""")
code(r"""# 1. Configuration  -- adapt the paths to your machine
""" + net_config + r"""STRAIN_DIR   = r"C:\Users\Tandale\sciebo\olddata\Phd\Writing\SLMU_paper\BVPs\Stress-Strain_Data_csv\BVP4_100_els"   # strain<element><l><m><n>.csv
ELEMENT      = """ + os.environ.get('GEN_ELEMENT', '1') + r"""         # element of the BVP whose strain files are used (1...100), e.g. 44 -> strain44<l><m><n>.csv

# Integration points (l, m, n) to integrate; all 45 of an element: [(l, m, n) for l in range(3) for m in range(3) for n in range(5)]
POINTS       = [(0, 0, 0)]
INCREMENTS   = """ + os.environ.get('GEN_INCREMENTS', '30') + r"""      # time increments (the strain files hold 30 rows; the original notebook ran 15)
STRAIN_PATH  = '""" + os.environ.get('GEN_STRAIN_PATH', 'file') + r"""'    # 'file': actual increment of step t = row t - row t-1 of strain1<l><m><n>.csv (cumulative strains,
                         #         cyclic: loading rows 0-9, load reversal from row 10)
                         # 'constant': row 0 at every step (as in the original notebook -> monotonic constant-increment loading)
LR           = 0.003     # Adam learning rate of the online optimisation (as in the original)
USE_XLA      = """ + os.environ.get('GEN_USE_XLA', 'True') + r"""      # compile the optimisation loop with XLA (tf.function(jit_compile=True))
ADAM_RESET   = '""" + os.environ.get('GEN_ADAM_RESET', 'never') + r"""'   # when Adam's moments/step counter are reset: 'solve' (every solve, as in the original),
                         # 'increment' (first solve of each increment), 'never' (only at the start of a run)
TOL          = """ + os.environ.get('GEN_TOL', '1e-9') + r"""      # convergence of the NN solve: residual**2 < TOL  (original notebook: 1e-7, i.e. |r| < 3.2e-4 MPa;
                         # 1e-9 -> |r| < 3.2e-5 MPa; the classical solver stops at |r| < 1e-7 MPa)
MAX_ITER     = 5000      # max optimiser steps per solve, the same for every solve: XLA compiles once per distinct value
                         # (the original used 5000 in the first and 1000 in later plane-stress iterations -> 2 compiles)""")

md("# 1. Network: frozen feature extractor + trainable last layer")
if ARCH == 'simple_rnn':
    code("""tf.random.set_seed(SEED)

def build_feature_model(units):
    \"\"\"2 x SimpleRNN(units), random init, frozen; returns features (last hidden output) and the new states.\"\"\"
    layer1 = tf.keras.layers.SimpleRNN(units, return_sequences=True, return_state=True)
    layer2 = tf.keras.layers.SimpleRNN(units, return_sequences=True, return_state=True)
    inputs = Input((None, 12))
    states = [Input(shape=(units,)) for _ in range(2)]
    X,  h1 = layer1(inputs, initial_state=states[0])
    X2, h2 = layer2(X,      initial_state=states[1])
    m = Model([inputs] + states, [X2[:, -1, :], h1, h2])
    m.trainable = False
    return m

feature_model = build_feature_model(UNITS)
STATE_KEYS    = ('state_h_', 'state_h1_')           # recurrent states stored per Gauss point in mat_params
STATE_UNITS   = UNITS

# trainable last layer Dense(1): random init (Glorot uniform, zero bias, like keras Dense), float64
W_init = tf.keras.initializers.GlorotUniform(seed=SEED)((UNITS, 1)).numpy().astype(np.float64)
B_init = np.zeros((1,), dtype=np.float64)
W = tf.Variable(W_init, name='W_last')   # (UNITS, 1)
B = tf.Variable(B_init, name='b_last')   # (1,)

def reset_last_layer():
    W.assign(W_init); B.assign(B_init)

print(f'trainable parameters: {W.shape[0] + 1}, frozen parameters: {feature_model.count_params()}')""")
else:
  code("""with open(WEIGHTS_PATH, "rb") as f:
    loaded_weights = pickle.load(f)

def build_feature_model(w):
    \"\"\"Same architecture / weights as model_klassiche_up in the original notebook, cut before Dense(1).\"\"\"
    layer1 = tf.keras.layers.RNN(HSNN(208, thr=1e-7), return_sequences=True, return_state=True)
    layer2 = tf.keras.layers.RNN(HSNN(208, thr=1e-7), return_sequences=True, return_state=True)
    layer3 = tf.keras.layers.RNN(HSNN(208, thr=1e-7), return_sequences=True, return_state=True)
    layer5 = tf.keras.layers.RNN(DSNN(208, thr=1e-7), return_sequences=True, return_state=True)
    layer7 = Dense(240, activation='tanh')
    layer8 = Dense(64, activation='tanh')
    layer1.build((None, None, 12));  layer1.set_weights(w[0:2])
    layer2.build((None, None, 208)); layer2.set_weights(w[2:4])
    layer3.build((None, None, 208)); layer3.set_weights(w[4:6])
    layer5.build((None, None, 208)); layer5.set_weights(w[6:8])
    layer7.build((None, None, 208)); layer7.set_weights(w[8:10])
    layer8.build((None, 240));       layer8.set_weights(w[10:12])

    inputs = Input((None, 12))
    states = [Input(shape=(208,)) for _ in range(8)]
    X,  h1, c1 = layer1(inputs, initial_state=states[0:2])
    X2, h2, c2 = layer2(X,      initial_state=states[2:4])
    X3, h3, c3 = layer3(X2,     initial_state=states[4:6])
    V,  h4, c4 = layer5(X3,     initial_state=states[6:8])
    feats      = layer8(layer7(tf.math.tanh(V)))
    m = Model([inputs] + states, [feats, h1, c1, h2, c2, h3, c3, h4, c4])
    m.trainable = False
    return m

feature_model = build_feature_model(loaded_weights)
STATE_KEYS    = ('state_h_', 'state_c_', 'state_h1_', 'state_c1_', 'state_h2_', 'state_c2_', 'state_h3_', 'state_c3_')
STATE_UNITS   = 208

# trainable last layer Dense(1), initialised with the meta-pretrained weights (float64)
W = tf.Variable(loaded_weights[12].astype(np.float64), name='W_last')   # (64, 1)
B = tf.Variable(loaded_weights[13].astype(np.float64), name='b_last')   # (1,)

def reset_last_layer():
    W.assign(loaded_weights[12].astype(np.float64)); B.assign(loaded_weights[13].astype(np.float64))""")

md("# 2. Material model parameters and helper functions (unchanged from the original notebook)")
code(material_cell)

md("# 3. Classical solver (copied unchanged from `Klassich_implicit.ipynb`)")
code(classical_cell)

md("# 4. Last-layer online optimisation as one compiled graph")
code("""f64 = lambda v: tf.constant(v, dtype=tf.float64)

def consistency_residual(D, sdev, Xn, mu, sy, K, R, b1, b2, a_unit, s_unit, N, dt):
    \"\"\"Same residual RHS_N as train_LSTM_static_KI_ in the original notebook, in TF ops.\"\"\"
    h     = sy + K * (D / dt) ** (1 / N) + (R + b1 * b2 * D) / (1 + b1 * D)
    Z     = sdev - Xn / (1 + s_unit * D)
    alpha = 1 + (3 * mu + a_unit / (1 + s_unit * D)) * (D / h)
    J     = tf.sqrt(1.5 * tf.reduce_sum(Z * Z))
    return J / alpha - sy - K * (D / dt) ** (1 / N) - (R - b1 * b2 * D) / (1 + b1 * D)

# Adam state kept OUTSIDE the solve so it can persist between solves (see ADAM_RESET)
mW_s, vW_s = tf.Variable(tf.zeros_like(W)), tf.Variable(tf.zeros_like(W))
mB_s, vB_s = tf.Variable(tf.zeros_like(B)), tf.Variable(tf.zeros_like(B))
step_s     = tf.Variable(f64(0.0))                                  # Adam step counter (bias correction)

def reset_adam():
    for v in (mW_s, vW_s, mB_s, vB_s):
        v.assign(tf.zeros_like(v))
    step_s.assign(0.0)

def pack_params(mat_params, key, stress_dev_trial, max_val, reset):
    \"\"\"All float inputs of one solve in ONE float64 vector (one host->graph transfer instead of 17).\"\"\"
    s = [mat_params['mu'+key], mat_params['stress_yield'+key], mat_params['stress_K'+key], mat_params['R'+key],
         mat_params['b1'+key], mat_params['b2'+key], mat_params['a_unit'+key], mat_params['s_unit'+key],
         N, timeincrement, max_val, LR, TOL, float(reset)]
    return tf.constant(np.concatenate([np.asarray(s, dtype=np.float64), np.ravel(stress_dev_trial),
                                       np.ravel(mat_params['stress_back_n-1'+key])]))

@tf.function(jit_compile=USE_XLA)
def fused_solve(x, states, pv, max_iter):
    \"\"\"ONE compiled (XLA) call per solve: frozen network forward pass -> features + new recurrent states, then
    Adam on the last layer (W, B) until residual**2 < tol, continuing from the stored Adam state unless reset.\"\"\"
    out = feature_model([x] + states, training=False)
    phi = tf.reshape(tf.cast(out[0], tf.float64), (1, -1))
    mu, sy, K, R, b1, b2, a_unit, s_unit, N_, dt, max_val, lr, tol, reset = tf.unstack(pv[:14])
    sdev, Xn = tf.reshape(pv[14:23], (3, 3)), tf.reshape(pv[23:32], (3, 3))
    beta1, beta2, eps = f64(0.9), f64(0.999), f64(1e-7)          # Keras Adam defaults
    keep = 1.0 - reset                                            # reset == 1 -> start from zero moments / step 0
    mW, vW, mB, vB, step = keep * mW_s, keep * vW_s, keep * mB_s, keep * vB_s, keep * step_s
    D, loss, iters = f64(0.0), f64(np.inf), tf.constant(0)
    for k in tf.range(max_iter):
        with tf.GradientTape() as tape:
            D    = max_val * tf.sigmoid(tf.matmul(phi, W)[0, 0] + B[0])
            r    = consistency_residual(D, sdev, Xn, mu, sy, K, R, b1, b2, a_unit, s_unit, N_, dt)
            loss = r * r
        iters = k + 1
        if loss < tol:
            break
        gW, gB = tape.gradient(loss, [W, B])
        step   = step + 1
        mW = beta1 * mW + (1 - beta1) * gW;  vW = beta2 * vW + (1 - beta2) * gW * gW
        mB = beta1 * mB + (1 - beta1) * gB;  vB = beta2 * vB + (1 - beta2) * gB * gB
        lr_t = lr * tf.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
        W.assign_sub(lr_t * mW / (tf.sqrt(vW) + eps))
        B.assign_sub(lr_t * mB / (tf.sqrt(vB) + eps))
    mW_s.assign(mW); vW_s.assign(vW); mB_s.assign(mB); vB_s.assign(vB); step_s.assign(step)
    return D, loss, iters, out[1:]


def upper_bound(mat_params, key, stress_eqv_trial, stress_back_trial, stress_dev_trial):
    \"\"\"max_val of the scaled sigmoid, identical to train_LSTM_static_KI_ in the original notebook.\"\"\"
    Delta_pl_U1 = np.maximum(0, (((stress_eqv_trial - mat_params['stress_yield'+key]) / mat_params['stress_K'+key]) ** (N)) * timeincrement)
    stress_tens_n1     = voigt_to_tens(mat_params['stress_n-1'+key])
    stress_dev_trial_n = stress_tens_n1 - I * (1/3 * stress_tens_n1.trace())
    stress_eqv_trial_n = np.sqrt(1.5 * (np.tensordot(stress_dev_trial_n - stress_back_trial, stress_dev_trial - stress_back_trial)))
    de = voigt_to_tens(mat_params['delta_eps'+key])
    Delta_pl_U2 = np.sqrt((2/3) * np.tensordot(de, de)) + (stress_eqv_trial_n - mat_params['stress_yield'+key]) / mat_params['E'+key]
    return Delta_pl_U2 if Delta_pl_U2 > Delta_pl_U1 == 0 else Delta_pl_U1


def flow_quantities(D, mat_params, key, stress_dev_trial):
    \"\"\"h_N, Z_N, alpha_N, J_N at D (numpy, float64); the original uses N_plas = Z_N / h_N.\"\"\"
    sy, K = mat_params['stress_yield'+key], mat_params['stress_K'+key]
    R, b1, b2 = mat_params['R'+key], mat_params['b1'+key], mat_params['b2'+key]
    h = sy + K * (D / timeincrement) ** (1 / N) + (R + b1 * b2 * D) / (1 + b1 * D)
    Z = stress_dev_trial - mat_params['stress_back_n-1'+key] / (1 + mat_params['s_unit'+key] * D)
    alpha = 1 + (3 * mat_params['mu'+key] + mat_params['a_unit'+key] / (1 + mat_params['s_unit'+key] * D)) * (D / h)
    J = np.sqrt(1.5 * np.tensordot(Z, Z))
    return h, Z, alpha, J


def local_solve(solver, flow, mat_params, key, l, m, n, stress_dev_trial, stress_back_trial, stress_eqv_trial, state_inputs, max_iter, log, t, p):
    \"\"\"One return-mapping solve for Delta_pl. Returns Delta_pl, N_plas, new recurrent states.\"\"\"
    rec = dict(t=t + 1, l=l, m=m, n=n, p=p + 1)

    tc = time.perf_counter()
    D_cl, *_ = implicit_integration_vp_gnl_new(mat_params, i, [], [], N, timeincrement, 0, stress_eqv_trial,
                                               stress_back_trial, stress_dev_trial, l, m, n)
    rec.update(D_classical=float(D_cl), time_classical=time.perf_counter() - tc)

    new_states = state_inputs
    if solver == 'nn':
        reset = ADAM_RESET == 'solve' or (ADAM_RESET == 'increment' and p == 0)
        tn = time.perf_counter()
        x = gen_input(stress_dev_trial, stress_back_trial)
        max_val = upper_bound(mat_params, key, stress_eqv_trial, stress_back_trial, stress_dev_trial)
        D, loss, iters, new_states = fused_solve(x, state_inputs, pack_params(mat_params, key, stress_dev_trial, max_val, reset),
                                                 tf.constant(max_iter))
        new_states = list(new_states)
        Delta_pl = float(D.numpy())
        rec.update(D_nn=Delta_pl, loss=float(loss.numpy()), iters=int(iters.numpy()), max_val=float(max_val),
                   converged=bool(loss.numpy() < TOL), time_nn=time.perf_counter() - tn)
    else:
        Delta_pl = float(D_cl)

    h_N, Z_N, alpha_N, J_N = flow_quantities(Delta_pl, mat_params, key, stress_dev_trial)
    rec['alpha'] = float(alpha_N)
    log.append(rec)
    # flow direction: 'Z/h' as in the self-learning notebook, 'Z/J' as in Klassich_implicit.ipynb (they differ by alpha_N)
    return Delta_pl, (Z_N / h_N if flow == 'Z/h' else Z_N / J_N), new_states""")

md("# 5. Integration loop (same algorithm as cell 'Test von Integration Schema' of the original)")
code("""mat_params_initial = copy.deepcopy(mat_params)
i = 0

def run(solver, flow='Z/J'):
    \"\"\"solver = 'nn' (last-layer self-learning) or 'classical'; flow = 'Z/J' (Klassich_implicit, default) or 'Z/h' (original self-learning notebook).\"\"\"
    mp  = copy.deepcopy(mat_params_initial)
    reset_adam()                                 # every run starts with a fresh optimiser state
    for (l, m, n) in POINTS:                    # zero recurrent states sized for the chosen network
        for s in STATE_KEYS:
            mp[s + str(i+1) + str(l+1) + str(m+1) + str(n+1)] = tf.zeros(shape=(1, STATE_UNITS))
    log = []
    # cumulative total strains per point (rows = time steps); increment of step t = row t - row t-1
    strain_rows = {(l, m, n): np.genfromtxt(os.path.join(STRAIN_DIR, 'strain' + str(ELEMENT) + str(l+1) + str(m+1) + str(n+1) + '.csv'))
                   for (l, m, n) in POINTS}
    if STRAIN_PATH == 'file':
        assert all(INCREMENTS <= len(v) for v in strain_rows.values()), 'INCREMENTS exceeds the rows of the strain file'
    for t in range(INCREMENTS):
        for (l, m, n) in POINTS:
            key  = str(i+1) + str(l+1) + str(m+1) + str(n+1)
            rows = strain_rows[(l, m, n)]
            if STRAIN_PATH == 'file':
                mp['delta_eps'+key]   = rows[t] - rows[t-1] if t > 0 else rows[0].copy()
            else:
                mp['delta_eps'+key]   = rows[0].copy()
            mp['stress_n-1'+key]      = mp['stress_n'+key].copy()
            sig_tr_i_1                = mp['stress_n'+key] + mp['C'+key].dot(mp['delta_eps'+key])
            mp['stress_back_n-1'+key] = mp['stress_back'+key].copy()
            stress_back_trial         = mp['stress_back'+key].copy()
            stress_dev_trial, _       = split(sig_tr_i_1)
            stress_eqv_trial          = np.sqrt(1.5 * (np.tensordot(stress_dev_trial - stress_back_trial, stress_dev_trial - stress_back_trial)))

            if stress_eqv_trial > abs(mp['stress_yield'+key]):
                state_inputs = [mp[s + key] for s in STATE_KEYS]
                for p in range(100):
                    if p != 0:
                        sig_tr_i_1       = mp['stress_n-1'+key] + mp['C'+key].dot(mp['delta_eps'+key])
                        stress_dev_trial, _ = split(sig_tr_i_1)
                        stress_eqv_trial = np.sqrt(1.5 * (np.tensordot(stress_dev_trial - stress_back_trial, stress_dev_trial - stress_back_trial)))
                        sigma_33_trial   = sig_tr_i_1[2]
                        Delta_pl, N_plas, new_states = local_solve(solver, flow, mp, key, l, m, n, stress_dev_trial, stress_back_trial,
                                                                   stress_eqv_trial, state_inputs, MAX_ITER, log, t, p)
                        sigma_i_1             = sigma_33_trial - (3 * mp['mu'+key] * Delta_pl) * N_plas[2][2]
                        delta_eps33_i_1       = delta_eps33_i_minus_1 - ((sigma_i * (delta_eps33_i - delta_eps33_i_minus_1)) / (1e-7 + sigma_i_1 - sigma_i))
                        mp['delta_eps'+key][2] = delta_eps33_i_1
                        delta_eps33_i_minus_1 = delta_eps33_i
                        delta_eps33_i         = delta_eps33_i_1
                        sigma_i               = sigma_i_1
                        if abs(delta_eps33_i - delta_eps33_i_minus_1) / abs(delta_eps33_i_1) < 1e-5:
                            break
                    else:
                        sigma_33_trial = sig_tr_i_1[2]
                        Delta_pl, N_plas, new_states = local_solve(solver, flow, mp, key, l, m, n, stress_dev_trial, stress_back_trial,
                                                                   stress_eqv_trial, state_inputs, MAX_ITER, log, t, p)
                        sigma_i_1             = sigma_33_trial - (3 * mp['mu'+key] * Delta_pl) * N_plas[2][2]
                        sigma_i               = sigma_i_1
                        delta_eps33_i_minus_1 = mp['delta_eps'+key][2]
                        mp['delta_eps'+key][2] = -(mp['delta_eps'+key][0] + mp['delta_eps'+key][1])
                        delta_eps33_i         = mp['delta_eps'+key][2]

                for s, v in zip(STATE_KEYS, new_states):
                    mp[s + key] = v

                mp['eps_el_tens'+key] = mp['eps_el_tens'+key] + voigt_to_tens(mp['delta_eps'+key])
                a1 = tens_to_voigt(mp['eps_pl_tens'+key])
                b1 = tens_to_voigt(mp['eps_el_tens'+key])
                c1 = tens_to_voigt(N_plas)
                for j in range(3):
                    a1[j] = a1[j] + (1.5 * Delta_pl * c1[j]);  b1[j] = b1[j] - (1.5 * Delta_pl * c1[j])
                for j in range(3, 6):
                    a1[j] = a1[j] + (3 * Delta_pl * c1[j]);    b1[j] = b1[j] - (3 * Delta_pl * c1[j])
                mp['eps_pl_tens'+key]      = voigt_to_tens(a1)
                mp['eps_el_tens'+key]      = voigt_to_tens(b1)
                mp['EQPLAS'+key]           = mp['EQPLAS'+key] + Delta_pl
                mp['flow'+key]             = tens_to_voigt(N_plas)
                mp['stress_n'+key]         = mp['C'+key].dot(mp['delta_eps'+key]) - (Delta_pl * mp['flow'+key]) * 3 * mp['mu'+key] + mp['stress_n-1'+key]
                mp['stress_back'+key]      = (mp['stress_back'+key] + (mp['a_unit'+key] * Delta_pl * N_plas)) / (1 + mp['s_unit'+key] * Delta_pl)
                mp['stress_eqv_trial'+key] = stress_eqv_trial
                mp['abs_strain'+key].append(tens_to_voigt(mp['eps_el_tens'+key] + mp['eps_pl_tens'+key]))
                mp['abs_stress'+key].append(mp['stress_n'+key])
                mp['Delta_pl'+key]         = Delta_pl
                mp['stress_back'+key][2][2] = 0
            else:
                mp['eps_el_tens'+key]      = mp['eps_el_tens'+key] + voigt_to_tens(mp['delta_eps'+key])
                mp['stress_n'+key]         = mp['stress_n'+key] + mp['C'+key].dot(mp['delta_eps'+key])
                mp['stress_eqv_trial'+key] = stress_eqv_trial
                mp['C_ep'+key]             = mp['C'+key].copy()
                mp['abs_strain'+key].append(tens_to_voigt(mp['eps_el_tens'+key] + mp['eps_pl_tens'+key]))
                mp['abs_stress'+key].append(mp['stress_n'+key])
    return mp, log""")

code("""reset_last_layer()
_t0 = time.perf_counter(); mp_nn, log_nn = run('nn');               time_run_nn = time.perf_counter() - _t0
_t0 = time.perf_counter(); mp_cl, log_cl = run('classical');        time_run_cl = time.perf_counter() - _t0
mp_ch, log_ch = run('classical', flow='Z/h')   # original self-learning formulation Z/h, for reference
print(f'settings            : USE_XLA={USE_XLA}, ADAM_RESET={ADAM_RESET!r}, LR={LR}, INCREMENTS={INCREMENTS}, STRAIN_PATH={STRAIN_PATH!r}, TOL={TOL}')
print(f'strain files        : element {ELEMENT}, points ' + ', '.join(f'strain{ELEMENT}{l+1}{m+1}{n+1}.csv' for (l, m, n) in POINTS))
print(f'NN (last layer) run : {time_run_nn:8.2f} s  ({len(log_nn)} solves)')
print(f'Classical run       : {time_run_cl:8.2f} s  ({len(log_cl)} solves)')""")

md("# 6. Comparison NN (last layer) vs classical")
code("""import pandas as pd
df = pd.DataFrame(log_nn)
df['rel_err'] = (df.D_nn - df.D_classical).abs() / df.D_classical.abs()
print('solves                    :', len(df))
print('converged (res^2 < TOL)   :', int(df.converged.sum()), '/', len(df))
print('optimiser steps  mean/max :', round(df.iters.mean(), 1), '/', df.iters.max())
print('rel. error Δε_p  max      : %.2e' % df.rel_err.max())
# Timing per solve: time_nn and time_classical are measured on the SAME solve (same trial state) in the NN run.
# The first NN call traces the graph and compiles it with XLA -> excluded from the comparison for BOTH solvers.
df['speedup'] = df.time_classical / df.time_nn          # > 1: NN faster on this solve
cmp = df.iloc[1:]
print('compile call (1st NN solve)          : %.0f ms (excluded below)' % (1e3 * df.time_nn.iloc[0]))
slow = cmp.index[cmp.time_nn > 0.05]
print('other NN calls > 50 ms (recompiles?) :', 'none' if len(slow) == 0 else list(slow))
print('time per solve (same %d solves)       : NN median %.3f ms / mean %.3f ms   classical median %.3f ms / mean %.3f ms'
      % (len(cmp), 1e3 * cmp.time_nn.median(), 1e3 * cmp.time_nn.mean(),
         1e3 * cmp.time_classical.median(), 1e3 * cmp.time_classical.mean()))
print('per-solve ratio t_classical / t_NN    : median %.2f, min %.2f, max %.2f; NN faster in %d of %d solves'
      % (cmp.speedup.median(), cmp.speedup.min(), cmp.speedup.max(), int((cmp.speedup > 1).sum()), len(cmp)))
for (l, m, n) in POINTS:
    key = str(i+1) + str(l+1) + str(m+1) + str(n+1)
    s_nn, s_cl = np.array(mp_nn['abs_stress'+key]), np.array(mp_cl['abs_stress'+key])
    s_ch = np.array(mp_ch['abs_stress'+key])
    print(f'point {key}: NN vs classical (both flow Z/J): max |Δσ| = {np.abs(s_nn - s_cl).max():.3e} MPa'
          f' (max |σ| = {np.abs(s_cl).max():.1f} MPa), EQPLAS {mp_nn["EQPLAS"+key]:.6e} vs {mp_cl["EQPLAS"+key]:.6e}')
    print(f'point {key}: original Z/h vs Z/J (classical solver both): max |Δσ| = {np.abs(s_ch - s_cl).max():.3e} MPa,'
          f' EQPLAS {mp_ch["EQPLAS"+key]:.6e} vs {mp_cl["EQPLAS"+key]:.6e}')
print('alpha_N at the solves     : min %.4f  max %.4f' % (df.alpha.min(), df.alpha.max()))
pd.set_option('display.max_rows', 500)
df[['t', 'p', 'D_nn', 'D_classical', 'rel_err', 'loss', 'iters', 'time_nn', 'time_classical', 'speedup']]""")

code("""fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for (l, m, n) in POINTS:
    key = str(i+1) + str(l+1) + str(m+1) + str(n+1)
    e_nn, s_nn = np.array(mp_nn['abs_strain'+key]), np.array(mp_nn['abs_stress'+key])
    e_cl, s_cl = np.array(mp_cl['abs_strain'+key]), np.array(mp_cl['abs_stress'+key])
    axs[0].plot(e_cl[:, 0], s_cl[:, 0], 'k-', label=f'classical {key}')
    axs[0].plot(e_nn[:, 0], s_nn[:, 0], 'o', mfc='none', label=f'NN last layer {key}')
    e_ch, s_ch = np.array(mp_ch['abs_strain'+key]), np.array(mp_ch['abs_stress'+key])
    axs[0].plot(e_ch[:, 0], s_ch[:, 0], 'r--', label=f'classical, original flow Z/h {key}')
axs[0].set_xlabel('ε11'); axs[0].set_ylabel('σ11 [MPa]'); axs[0].legend(); axs[0].grid()
axs[1].semilogy(1e3 * df.time_nn.values[1:], label='NN last layer (graph)')
axs[1].semilogy(1e3 * df.time_classical.values[1:], label='classical')
axs[1].set_xlabel('solve #'); axs[1].set_ylabel('time per solve [ms]'); axs[1].legend(); axs[1].grid()
plt.tight_layout(); plt.show()""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python (self_learning)', language='python', name='self_learning'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
for c in nb['cells']:
    c['source'] = c['source']
out = sys.argv[1]
json.dump(nb, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('written', out)
