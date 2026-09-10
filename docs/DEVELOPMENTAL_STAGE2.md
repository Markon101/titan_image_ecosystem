# Second-stage frozen dynamical analysis

This extends `titan_develop`; training/model/checkpoint behavior is unchanged.
Use `--extended true` for bounded extra exports, and `--lyap-vectors 3` for the
central-difference QR ensemble. All inputs are frozen, target references absent,
and clock sequences shared. Existing strict import, create-new output directories,
source hashes and completion receipts remain mandatory. New exports are versioned
`titan.development.v2` and hashed by the final summary. Diagnostics can run from
any compatible saved age; ages in response arguments are offsets from that state.

```sh
cargo build --release --locked --bin titan_develop
./target/release/titan_develop --config CONFIG.json --output NEW_DIRECTORY \
  --steps 64 --extended true --lyap-vectors 3 --lyap-epsilon .16 --qr-every 4 \
  --response-ages 0,16,32,64 --response-bands low,mid,high \
  --response-epsilons .02,.04,.08,.16,.32,.64,1.28,2.56
python scripts/development_analysis.py --input NEW_DIRECTORY --output NEW_POST_DIRECTORY
```

NumPy is required only for offline analysis. The Rust binary has no new dependency.
`--event-channel` selects a micro channel (default 0). Extended export is limited
to 4096 steps, QR to four vectors, response panels to eight amplitudes and 64 ages.
Recurrence retention retains its 512 MiB ceiling. Do not infer total-process memory
from that ceiling: QR adds up to eight extra worlds and f64 QR scratch.

## Precision and local Jacobian action

The symmetric numerator is evaluated by converting **f32 outputs to f64 before
addition/subtraction**. `response.jsonl` saves both `N_full=||G+ + G- - 2G||`
(including memory) and `N_low=||LP(G+ + G- - 2G)||` (spatial fields only), plus
`Q=N_low/(2 epsilon²)`. Zero-mean perturbations use the original Fourier bands
and one deterministic unit-L2 micro direction per band, reused across ages and
amplitudes. Input f32 asymmetry and actual amplitudes remain recorded.

Offline local slopes are `log(value_b/value_a)/log(epsilon_b/epsilon_a)`.
A resolved-window candidate needs two adjacent intervals/three amplitudes with
full and low numerator slopes within .35 of 2 and Q slopes within .35 of 0.
Noise-like intervals have low-numerator slope within .35 of 0 and Q slope within
.35 of -2. Thresholds are explicit numerical heuristics, not confidence intervals.
An empty window list is an unresolved result, not absence of nonlinearity.

The same paired evaluations provide `Jv ≈ (G+ - G-)/(2 epsilon)` in full-state
coordinates. `local_jacobian.jsonl` stores the tiny Gram matrix of output vectors.
Eigenvalues of that Gram matrix yield squared singular gains **restricted to the
supplied orthonormal band directions**, not the global extrema of J. Coefficients
of the most/least amplified combinations and each direction's susceptibility are
exported offline. Compare across epsilon to distinguish numerical resolution from
finite-amplitude nonlinear contamination. The model still computes in f32; f64
accumulation does not make this a double-precision Titan trajectory.

## Finite-time spectrum approximation

Initialize k full-state random vectors with seeded ChaCha8 noise, then twice apply
modified Gram-Schmidt to obtain an orthonormal basis Q. For each column q evolve
`x+epsilon*q` and `x-epsilon*q` under identical inputs/clocks for `qr_every` steps.
Form columns `(x_plus-x_minus)/(2 epsilon)`, QR-reorthogonalize in f64, accumulate
`log(R_ii)`, and reset central pairs around the actual baseline trajectory.
`lambda_i(T)=sum_blocks log(R_ii)/T`; the final partial block is included.
`lyapunov.jsonl` saves each block stretch, cumulative exponent, orthogonality error,
f32 reset relative error, and central-pair midpoint drift divided by epsilon.
Rank loss or >1% reset distortion fails the run; no exponent flooring is applied.

These are conditional, finite-amplitude QR estimates. There is no tangent-space
warm-up by default. With few vectors they approach leading exponents only with
sufficient alignment time; ordering and convergence are not assumed. Classification
uses the sampled exponents: positive/negative tolerance .01 per step, and all
exponents below -.05 for strongly contractive. Mixed means both signs outside
that tolerance; near-critical means none above .01 and at least one within the
neutral interval. These describe sampled directions, not global stability or chaos.

## Recurrence quantification and DMD

Distances are divided by sqrt(full-state scalar count), making epsilon a state-RMS
threshold. Defaults `.01,.05,.1,.2` are shared across runs. Exclude pairs whose age
difference is <=2 steps (Theiler window). RR is recurrent/eligible pairs. DET is
the fraction of recurrent points in diagonal runs of at least two samples; LAM is
the corresponding vertical fraction. Average/max diagonal length and mean qualifying
vertical length (trapping time) use maximal runs, including edge-truncated ones.
Return-time histograms count gaps between starts of **forward-time** recurrent
runs after each reference age plus the Theiler window, in developmental steps.
This avoids counting an artificial split across the excluded diagonal as a return.
The corrected offline schema is `titan.dynamical_analysis.v2.1`. Empty point/line denominators produce null.
Line lengths are retained samples, not steps; keep recurrence stride fixed across
comparisons. Recurrence does not imply periodicity; cadence can create diagonal lines.

DMD uses 6 spectral energies, two variances, memory norm, update norm, 16 fixed
Rademacher state projections and memory mean/std. Random projection coefficients
are deterministic SplitMix64 signs scaled by `1/sqrt(n)`; no PCA of full state is
claimed. Input/output training means and input standard deviations are fitted only
on the first 70% of chronological pairs. Near-constant features are removed; rank
is capped at six and singular values below 1e-8 of the largest are dropped.
For centered normalized X=UΣV*, fit B=YVΣ^-1, K=U*B. The affine full-observable
prediction is `mean_Y + B U*(z-mean_X)`; eig(K) describes the reduced fluctuations.

Reports include compression error, train and held-out one-step RMSE, a persistence
baseline, recursive held-out RMSE and fits to temporal halves. Eigenvalue labels
use radius .99/1.01 and nonzero imaginary part, but interpretation is supported only
if held-out prediction beats persistence. This is an observable model, not a global
Koopman spectrum or a model of latent-state reconstruction.

## Cross-scale work and events

For each field and band, `T_s=<P_s x_t,P_s delta_x_t>` uses the real spectral inner
product divided by HW. It satisfies the measured identity
`E_s(t+1)-E_s(t)=2 T_s+||P_s delta_x_t||²`.
Negative work can coexist with increasing band energy if update energy dominates.
The whole legacy+sidecar update is measured; this is not an isolated nonlinear
conservative transfer term. Exported budget residuals check the calculation.
Lagged Pearson correlations use **energy increments**, with positive lag meaning
first band precedes second. Correlation and net work do not establish a cascade.

`spatial.f32le` contains one original-grid micro channel per step, row-major,
little-endian f32; shapes/frame count/channel are in manifest config and trajectory.
It is a compact event-observable export, not a full checkpoint/trajectory tensor.
Burst detection selects local maxima of full update norm above median+3*1.4826*MAD,
with four-step separation and complete +/-4 windows. Zero MAD means no resolved
events. Event windows contain band energies, update norms and band-activity peak
ordering; tied/mixed peaks are not forced into a sequence.

For the chosen micro state channel, remove its spatial mean, take amplitude as
max absolute deviation, center at that peak, and ell as energy-weighted RMS periodic
radius. Radial bin averages on xi=r/ell are divided by amplitude. Pairwise RMSE of
common normalized-profile bins quantifies collapse; fewer than two valid events
leaves collapse unavailable. This radial, single-channel statistic does not resolve
anisotropy or full multichannel morphology. No burst is invented to fill the report.

## Phase signatures and limitations

At each QR block, export exponents, spatial spectral fractions, update norm and
prefix recurrence metrics for each epsilon. Parameter sweeps compare those measured
signatures across separate bounded sidecars and saved ages. All categorical labels
retain their sampled-subspace qualification. Do not read a parameter boundary into
one isolated sign change, or compare two differently trained checkpoints as a matched
architecture ablation. No trainable operator, objective, clamp or attractor correction
is introduced by this stage.

Method references: [Tu et al., DMD](https://arxiv.org/abs/1312.0041) motivates the
low-rank linear-map interpretation; [Recurrence Quantification of Fractal Structures](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2012.00382/full)
describes recurrence-line measures. Exact operational conventions above take
precedence when comparing these artifacts with other implementations.


## Corrected fixed-window map and precision control

After a panel completes, `scripts/development_window_map.py --panel PANEL --output NEW_DIR`
revalidates every raw run, applies corrected forward-return RQA, and saves JSON/CSV
phase tables. Its default trailing window is 16 steps. Window exponents are differences
of accumulated log stretches divided by elapsed window length; the basis remains
aligned from the original checkpoint. This separates late behavior from cumulative
initial transients without claiming independently initialized local leading exponents.
Original raw artifacts and earlier offline reports are preserved. Use this corrected
output for return-time distributions and the final phase comparison.

`scripts/development_precision_fixture.py --output NEW_FILE.json` compares f32 and
f64 evaluation of the analytic map `tanh(x)+.2*x²` on a 16x16 zero-mean checkerboard
perturbation. It checks the f64 Q limit against the analytic second derivative.
It is a diagnostic precision control, **not an f64 Titan implementation**.
