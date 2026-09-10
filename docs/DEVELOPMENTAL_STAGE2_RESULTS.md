# Frozen dynamical characterization: second-stage results

The epsilon sweeps resolve finite-amplitude quadratic-response windows in both
checkpoints. The selected RMSNorm trajectory exhibits transient perturbation
expansion followed by contraction. A matched, bounded constant-diffusion control
removes its positive full-horizon sampled QR growth. These observations support
one future learned-diffusion ablation; they do not establish chaos, an attractor
class, emergence, or better reconstruction/generalization.

No architecture, training objective, optimizer layout, checkpoint signature or
legacy state update was redesigned. The original user edits remain outside these
commits. All model evaluation is frozen CPU/f32 with reference tensors absent and
fidelity zero. Diagnostic arithmetic, QR, DFT, regression and accumulations use f64.

## First-stage completion and compatibility audit

The original baseline/zero/transport/diffusion experiments are complete and their
artifact/checkpoint hashes still verify. Baseline and zero-sidecar `trajectory.jsonl`
and `recurrence.f64le` are **byte-identical**, with SHA-256 respectively:

* `46865ef63905b874d3cf51c18e857268b7c3c74694a05f0eec55ff9407867630`
* `bd65767671f39dc5b4d57c0a3bd024c96da58585cb8ce492c0c6be9a507a3086`

First-stage commits:

* `fef12d8d72b88228607ea12754b069951561b92a`
* `ec58cb54c76461689c8c1f85354f72dff0091e8f`
* `2036306e586bae2bebcd653fdd2a8c78062ace8b`

Its original binary is preserved at `target/release/titan_develop_stage1_ec58cb5`,
SHA-256 `f8d1901970c0628f3c023b97562afe8d688d328b9709c1b6f6d4f12947057487`.
The four original total runtimes were 39.82/30.19/29.67/30.39 seconds, sampled peak
RSS about 311 MiB, and retained recurrence states 17.59 MiB. Details remain in
[the first-stage report](DEVELOPMENTAL_VALIDATION.md).

The new 64-step zero control also matches baseline bytes for trajectory,
observables, spatial fields, QR records and recurrence. The first nine state
records agree exactly with stage one after excluding the differently configured
perturbation diagnostic. All 14 existing files across the two checkpoint input
roots remain unchanged. `first_stage_audit.json` and `panel/protected_before.json`
under the evidence root preserve the audited identities.

## Inputs, definitions and reproducibility

Evidence root: `analysis/development_stage2_20260909/` (runs executed September 10).
The parent is step 122260, saved age 20, target index 41. RMSNorm is step 128196,
saved age 4, target index 88. Each was developed for 64 steps. Both have micro
`[1,32,80,80]`, macro `[1,32,40,40]`, memory `[1,160]`: 256160 state scalars.
These are distinct saved checkpoints/conditions, not a matched normalization
training ablation. The additional RMSNorm diffusion control is matched to RMSNorm.

Actual implemented definitions and every CLI option are in
[DEVELOPMENTAL_STAGE2.md](DEVELOPMENTAL_STAGE2.md). Key conventions:

* N is the full-state L2 symmetric numerator; N_low is its spatial low-pass L2.
  Q=N_low/(2 epsilon²). Slopes are adjacent log-log differences.
* Three full-state central perturbation pairs, epsilon .16, QR every four steps,
  twice-reorthogonalized modified Gram-Schmidt. Lambda=sum(log R_ii)/T. QR bases
  are carried from the checkpoint; these are sampled, finite-amplitude estimates.
* Recurrence uses full-state RMS distance, thresholds .01/.05/.1/.2, Theiler window
  two steps and minimum line length two samples. Return gaps are forward-only.
* DMD is a rank-at-most-six affine reduced map of standardized compact observables,
  with chronological 70% training pairs and held-out/persistence errors.
* Band work is <P_s x,P_s delta>, checked against delta E_s=2 work+||P_s delta||².
  Band cutoffs are radial .125/.25 cycles per original grid cell, with DC in low.
* Structured Jv=(G+−G−)/(2 epsilon); Gram eigenvalues yield singular gains only in
  three orthonormal micro-band directions, not global singular extrema.
* Burst threshold is median(update)+3*1.4826*MAD, local maxima, four-step spacing
  and complete +/-4 windows. Radial, single-channel normalized-profile RMSE is
  available when at least two events exist.
* Fixed-window phase estimates subtract cumulative log stretches across trailing
  16-step windows. Near-critical labels use a declared .01/step tolerance and
  describe sampled directions, not a proven critical point or bifurcation.

Checkpoint SHA-256 identities:

| Input | Component | SHA-256 |
|---|---|---|
| parent | checkpoint | `0927b2fba652674c3d0038ef1942f753f08e8f0844de005a4350767128b46a52` |
| parent | model | `4b7c713bb4eade6809909762fef5a7b747f662e84b62ad70bf433df94a392adf` |
| parent | optimizer | `c812156bbd9046c747b9cd12e3e41972d7e24023f9816bb9e06ce39882f7dfc1` |
| parent | world | `2dc7be88d10d1680863fd3c88f250ac357806e53b0c7b9a75eee1a276d1928d8` |
| rmsnorm | checkpoint | `ae02cb618fa422a5dbcd98b87fff4bb3c32df860147d109bc00ac39e1e95161b` |
| rmsnorm | model | `5fb7ec9303086b5b052be88bb6731ca146b67371fb2b6a50db8cead15c992074` |
| rmsnorm | optimizer | `4e0ed872baaebf799abc9616da44f87c0927b8fe021b292b28b8a43404e64eab` |
| rmsnorm | world | `6bd6c3e729d48494c486eb81475eb96a2a0982ad0549caef74976821ceb8928f` |

Full source/effective configs, seeds, coefficients, exact bands, clock policy,
training step, age, binary hash and load/optimizer report are in each manifest.
The clean tested Rust build is `6b8b3e427bd2e89c4541f7c23d7ea7e7a28e3b0d`.
Python panel implementation is `f6eed5326843baecda55f005107897764818afd5`;
corrected forward returns, fixed-window maps and precision validation are in
`57f404d3fdcc0c885b555244bcf702bbe4b45aa5`.

Current tested binary SHA-256: `5526ac73789ab737d1d4426f65991152b939102d810c2a3b5b4e33384a1ca792`.

## Numerically resolved nonlinear response

Eight amplitudes .02,.04,.08,.16,.32,.64,1.28,2.56 were tested at four offsets,
for each low/mid/high zero-mean direction. A window requires at least two adjacent
intervals with full and low numerator slopes within .35 of 2 and Q slope within
.35 of zero. These are operational numerical-resolution criteria, not confidence
intervals or proof of an infinitesimal Hessian limit.

| Checkpoint | Actual age | Low window | Mid window | High window |
|---|---:|---|---|---|
| parent | 20 | 0.16–2.56 | 0.08–2.56 | 0.08–2.56 |
| parent | 36 | 0.64–2.56 | 0.64–2.56 | 0.32–2.56 |
| parent | 52 | 0.64–2.56 | 0.64–2.56 | 0.64–2.56 |
| parent | 84 | 0.64–2.56 | 0.64–2.56 | 0.64–2.56 |
| rmsnorm | 4 | 0.04–2.56 | 0.04–2.56 | 0.04–2.56 |
| rmsnorm | 20 | 0.16–2.56 | 0.16–2.56 | 0.16–2.56 |
| rmsnorm | 36 | 0.32–2.56 | 0.16–2.56 | 0.16–2.56 |
| rmsnorm | 68 | 0.32–2.56 | 0.32–2.56 | 0.32–2.56 |

For parent age 20, low-band N_full changes only 3.246e-5→3.329e-5 when epsilon
.02→.04, while Q drops .008983→.002291: the subtraction-noise signature.
At epsilon .32→.64, N_full grows .0002490→.0009868, while Q remains
.0003606→.0003586: a resolved quadratic interval. The full sweep provides
neighboring intervals needed for the window criterion.

At epsilon 1.28, high-input Q drops across development from .001357 to .0001072
for parent and .003511 to .0002649 for RMSNorm. These are low-pass response norms,
including DC; even a local pointwise nonlinearity can produce such a response.
They establish measurable nonlinear response, not a missing stress mechanism,
conservative inter-band energy transfer, or a turbulence cascade.

`precision_fixture.json` compares the analytic CPU map tanh(x)+.2*x² in f32/f64.
The f64 result agrees with the analytic quadratic limit at epsilon .0001. It is
not an f64 Titan implementation. Model outputs remain f32; increasing accumulation
precision alone cannot recover a numerator already lost in model-output rounding.

## Finite-time growth and the phase map

| Run | Cumulative maximum sampled lambda | Last 16-step maximum | Final full energy/scalar | Final micro high energy |
|---|---:|---:|---:|---:|
| parent_baseline | -0.011267 | -0.007665 | 0.547775 | 7004.51 |
| parent_zero | -0.011267 | -0.007665 | 0.547775 | 7004.51 |
| parent_transport_0p005 | -0.014086 | -0.008072 | 0.542101 | 3807.44 |
| parent_transport_0p02 | -0.017544 | -0.007317 | 0.537181 | 1098.38 |
| parent_diffusion_0p005 | -0.015576 | -0.009537 | 0.541734 | 3621.88 |
| parent_diffusion_0p02 | -0.022855 | -0.012289 | 0.537044 | 1068.14 |
| rmsnorm_baseline | 0.013928 | -0.014676 | 0.403265 | 9750.96 |
| RMSNorm matched diffusion max=.02 | -0.015411 | -0.016842 | 0.386501 | 1312.58 |

Diffusion max .005/.02 means constant nu=.0025/.01 (zero sigmoid logits).
Transport bounds are per axis. Controls are separate, never combined.

RMSNorm window maxima over ages 4–20, 20–36, 36–52, 52–68 are respectively
+.02793, +.03346, +.01364, −.01468/step. Matched diffusion gives approximately
+.00033, −.01642, −.02512, −.01684. This supports an age-dependent change in
sampled transient response and an operator-dependent change in its sign.
It does not identify multiple asymptotic attractors or a bifurcation threshold.

Halving QR epsilon from .16 to .08 changes final parent exponents by at most
2.43e-8 and RMSNorm by 1.61e-5; maximum trailing-window differences are 5.78e-7
and 7.23e-5. QR orthogonality errors stay around 1e-12 and reset errors below
7e-5 relative L2. RMSNorm's maximum central midpoint drift/epsilon is .0173;
finite-amplitude bias is checked empirically by the epsilon repeat, not assumed zero.

The matched RMSNorm diffusion repeat at epsilon .08 agrees within 1.56e-06/step
in final exponents and 1.03e-05 across trailing windows; its trajectory records
and recurrence bytes are identical. The repeat took 184.4 seconds.

Parent absolute energy rises .13766→.54777 while its sampled perturbations
contract. RMSNorm energy rises .02445→.40327 even as its late window contracts.
Absolute growth and perturbation instability are distinct measured quantities.

Structured one-step micro gains illustrate the sampling limitation. At saved
RMSNorm age 4, the tested singular gains span .99793–1.01130; at age 68 they span
.96781–.96807. Parent spans .98546–.98840 initially and .95895–.95925 at age 84.
The three micro directions do not span all full-state expanding directions.
All structured age probes sample the same macro-cadence phase; four-step QR blocks
cover the entire cadence. Neither measurement is the full Jacobian spectrum.

## Recurrence, linear modes, transfer and events

| Run | RR at RMS .05 | RR at .1 | DET at .1 | LAM at .1 | Max diagonal at .1 |
|---|---:|---:|---:|---:|---:|
| parent_baseline | 0.1137 | 0.3001 | 0.9983 | 0.9983 | 62 |
| parent_transport_0p02 | 0.1229 | 0.3134 | 0.9984 | 0.9984 | 62 |
| parent_diffusion_0p02 | 0.1244 | 0.3159 | 0.9984 | 0.9984 | 62 |
| rmsnorm_baseline | 0.0522 | 0.1838 | 0.9972 | 0.9972 | 62 |

RR is zero at RMS .01 for these runs. High DET/LAM at larger thresholds coexist
with no repeated forward entry gaps for the main baseline comparisons. Slowly
moving nearby states produce long lines; these results do not establish true
periodicity. A single threshold-specific return in a transport control is not
robust cycle evidence. Average diagonal lengths, trapping times and every gap
histogram are in the corrected per-run JSON files.

The initial RQA implementation counted entries on both sides of the Theiler gap,
which could fabricate a return across the excluded diagonal. That was corrected
and regression-tested on monotonic and periodic fixtures. Raw runs and the original
panel reports were preserved; **use `corrected_panel/` for RQA and phase conclusions**.
The earlier panel's return-time histograms are superseded.

Every full-run DMD fit performs worse than held-out persistence; parent error is
.4855 versus .2458 standardized RMSE, with skill −.975. RMSNorm skill is −2.137.
Temporal-half fits also fail that comparison. Eigenvalue arrays and recursive
errors are exported, but their decaying/growing/oscillatory labels do not support
physical modal claims from these poor fits.

Band-work budgets close to about 3.3e-9 absolute error against energies of order
1e5. Parent and RMSNorm baseline have net positive work in all three bands;
lagged relationships are exported from energy increments. Net work includes the
entire update, and correlations do not isolate causal or scale-local transfer.

No run triggers the predeclared burst detector. There are therefore no real-event
collapse errors or repeatable L→M→H/H→M→L burst sequences to report. The profile
pipeline passes analytic amplitude/translation controls, but this does not prove
self-similarity in Titan. It may miss structural transitions with steady update
norm, and its single-channel radial profiles do not capture anisotropic boxes.

## Runtime, memory and validation

| Run | Total seconds | Base bucket | Diagnostics | Extra probes | Sampled peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| parent_baseline | 336.2 | 14.7 | 59.2 | 185.5 | 310.6 |
| parent_zero | 258.8 | 16.1 | 65.4 | 93.9 | 312.0 |
| parent_transport_0p005 | 253.4 | 16.5 | 67.3 | 97.4 | 311.4 |
| parent_transport_0p02 | 264.8 | 17.0 | 66.3 | 101.1 | 311.8 |
| parent_diffusion_0p005 | 270.3 | 17.5 | 69.6 | 104.4 | 311.6 |
| parent_diffusion_0p02 | 256.1 | 16.6 | 66.1 | 99.1 | 311.4 |
| rmsnorm_baseline | 375.1 | 16.6 | 67.7 | 209.3 | 336.6 |
| parent_epsilon_check | 261.5 | 16.6 | 67.4 | 98.0 | 311.3 |
| rmsnorm_epsilon_check | 242.6 | 14.6 | 63.3 | 87.2 | 311.7 |
| matched RMSNorm diffusion | 218.0 | 14.4 | 54.0 | 84.7 | 311.7 |

The nine-run panel took 2523 seconds (42.1 minutes), before the adaptive matched
control and its epsilon check. Startup/corpus/hashing is included only in total;
base includes host extraction/cancellation. QR/probe measurement overhead is spread
across the diagnostic and extra-probe buckets. This dependency-free DFT is expensive;
these numbers are measurements, not controlled hardware-throughput benchmarks.

Each 65-state recurrence history retained 133,203,200 bytes (127.03 MiB), the
65x65 matrix 33,800 bytes, and the single-channel spatial export 1,664,000 bytes.
Sampled process RSS includes startup/model/corpus/scratch and is not isolated
incremental memory overhead. All original source and checkpoint files remain read-only.

Validation: 11 focused Rust tests; seven Python analytical tests (including known
rotation/decay, recurrence, return-gap correction, QR windows, profile collapse and
noise/quadratic slopes); CPU f32/f64 analytic precision control; strict Clippy;
format/diff checks; optimized-Python hash rejection; exact zero-control output
comparisons; energy-budget checks; amplitude repeats; and artifact/source hash audits.
No checkpoint-model/optimizer migration or new trainable tensor was introduced.

## One justified next architecture experiment

Test **bounded learned diffusion only**, in a new controlled fork of the RMSNorm
checkpoint. The reason is specific: constant nu=.01 changes positive full-horizon
sampled QR growth to contraction and reduces final micro high-band energy by
86.5%, while micro low-band energy changes by only +.47%. It does not show a
reconstruction or morphology improvement; energy preservation is not shape preservation.

Compare fixed nu=.01 with a bounded state-dependent nu(x) in [0,.02], keeping
checkpoint initialization, seeds, clocks, training budget, existing objective and
other mechanisms matched. The experiment should test whether adaptive diffusion
can preserve more useful detail than constant damping while controlling transient
amplification. Evaluate held-out reconstruction and actual morphology alongside
QR/recurrence/band diagnostics. Keep the legacy branch and strict import/parity
checks intact. Do not add transport, stress, timescale groups or staged scales in
that experiment. No learned diffusion was implemented or trained in this stage.

## Handoff

* `panel/`: original nine frozen runs, exact commands and source hashes.
* `corrected_panel/`: corrected per-run RQA/DMD/response/events and
  `window_phase_map.json` / `.csv`, with immutable receipts.
* `adaptive_rmsnorm_diffusion/`: matched control, command, runtime and analysis.
* `precision_fixture.json`: analytic precision comparison.
* [Method/CLI documentation](DEVELOPMENTAL_STAGE2.md).

Stage-two source changes: `src/development/{main,spectral,ensemble,extended}.rs`,
`scripts/development_{analysis,phase_panel,precision_fixture,window_map}.py`,
`scripts/test_development_analysis.py`, `README.md`, `docs/DEVELOPMENTAL_STAGE2.md`
and this report. All new raw and derived numeric artifacts remain under ignored
`analysis/`; no large arrays or checkpoints were committed.
