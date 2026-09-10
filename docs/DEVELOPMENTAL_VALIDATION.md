# First-pass developmental validation, 2026-09-09

The frozen CPU implementation completed separate baseline, zero-sidecar,
transport-only and diffusion-only runs using an existing compatible v9 checkpoint.
All states and reported numeric values remained finite. This validates the
instrumentation and reachability of the controls, not learning gains or an
attractor classification. The small-epsilon response measurements are sensitive
to numerical precision and do not yet establish nonlinear mode coupling.

## Reproduce and inspect

```sh
python scripts/development_smoke.py \
  --config analysis/rmsnorm_followup_2026-09-09_matched/parent/config.json \
  --root analysis/development_2026-09-09/controlled
```

Use a new root when rerunning. The runner itself creates the four ablation
configurations, saves exact commands, verifies artifact hashes and recurrence
properties, samples process VmHWM, and compares baseline/zero-sidecar outputs.
Local evidence from this run is under
`analysis/development_2026-09-09/controlled/validation.json` and each run's
`manifest.json`, `summary.json`, `trajectory.jsonl`, `response.jsonl`,
`recurrence.json`, `recurrence.f64le`, and `temporal.json`.
These machine artifacts are intentionally outside Git under ignored `analysis/`.

The tested release binary is `target/release/titan_develop`, SHA-256
`f8d1901970c0628f3c023b97562afe8d688d328b9709c1b6f6d4f12947057487`.
It was built from a clean detached checkout of
`ec58cb54c76461689c8c1f85354f72dff0091e8f`, with the repository's release profile
and Rust flags `-C target-cpu=native -C target-feature=+fp16`, CPU backend.
The manifest records `build_dirty:false` (serialized as a string by the existing
build provenance API). The shared workspace had pre-existing uncommitted changes;
they were excluded from this build and from the new commits.

Implementation commits:

* `fef12d8d72b88228607ea12754b069951561b92a`: strict frozen CLI, diagnostics,
  original-grid DFT, recurrence and ±epsilon probe.
* `ec58cb54c76461689c8c1f85354f72dff0091e8f`: optional sidecar transport/diffusion
  and cancellation accounting.

## State, equations and configuration

Definitions and exact Fourier bands are in [DEVELOPMENTAL_DYNAMICS.md](DEVELOPMENTAL_DYNAMICS.md).
The checkpoint had training/world step **122260**, saved developmental age **20**,
checkpoint seed **42**, and was developed for **8** further steps to age **28**.
Micro shape was `[1,32,80,80]`, macro `[1,32,40,40]`, and hidden memory `[1,160]`:
**256160** full-state scalars. Reference tensors were absent; fidelity was zero.
Saved genome and deterministic clock sequence were identical across ablations.

Spatial energy is `sum(x²)/(2HW)`; combined energy below is explicitly per scalar,
`sum(full_state²)/(2*256160)`. Updates, perturbations and recurrence use full-state
unnormalized L2. Fourier bands are radial cell frequencies low `[0,.125]`,
mid `(.125,.25]`, high `>.25`, including DC only in low. The FT perturbation
was seeded high-band micro noise, global L2 epsilon `.02`; its actual f32 initial
L2 was `0.01999999059863283`.

Transport used max speed `.01` per axis, `ax=.01*tanh(channel0)` and
`ay=.01*tanh(channel1)`, periodic upwind differences. Diffusion used max `.01`
and zero-logit sigmoid, hence constant nu `.005`, with a five-point Laplacian.
They were tested individually. These are fixed controls, not trained projections.

## Observed results

| Run | Final full energy/scalar | Last update L2 | FT rate over 8 steps | Cancellation C | Initial-to-final recurrence L2 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.21999746 | 10.61061 | -0.010734 | 1.00000 | 79.69000 |
| Zero sidecar | 0.21999746 | 10.61061 | -0.010734 | 1.00000 | 79.69000 |
| Transport only | 0.21867594 | 10.49886 | -0.028181 | 1.11035 | 79.03736 |
| Diffusion only | 0.21834526 | 10.48943 | -0.034537 | 1.12200 | 78.92788 |

Baseline full-state energy/scalar rose from **0.13766443** to **0.21999746** while
the perturbation L2 fell from **0.0200000** to **0.0183542**. Increasing state
energy and local perturbation contraction coexist here. None of the four runs
met the configured state-stagnation criterion. Eight steps are insufficient for
periodic/quasiperiodic/metastability conclusions. No image was rendered, so no
visual change to nested-square behavior is claimed.

Final micro low/mid/high energies (unnormalized Parseval band sums):

| Run | Low | Mid | High |
|---|---:|---:|---:|
| Baseline | 106585.63 | 683.76 | 3155.39 |
| Transport | 106606.34 | 636.53 | 2508.61 |
| Diffusion | 106652.33 | 620.60 | 2315.78 |

The final transport norm was **1.01383** against aggregate legacy update norm
**10.64355**; diffusion norm was **1.11584** against legacy norm **10.65329**.
The f32 composition residual L2 was about **8.8e-6** for either active control.
C measures cancellation against the aggregate legacy delta; internal cancellation
among existing Titan operators remains unresolved. No new clamp was applied.
These are short numerical observations, not evidence of generalization or
long-horizon stability. Upwind transport itself has numerical diffusion.

## Nonlinear response and precision

`Q=LP(G(x+epsilon*p)+G(x-epsilon*p)-2G(x))/(2*epsilon²)` was tested in all three
bands at offsets 0 and 8, epsilons `.02` and `.04`, with the same unit-L2
perturbation direction for each age/band pair. For the high-band input:

| Age | Epsilon | Spatial Q L2 | Q / full baseline update L2 |
|---|---:|---:|---:|
| 20 | .02 | .00917597 | .00072311 |
| 20 | .04 | .00271708 | .00021412 |
| 28 | .02 | .01167461 | .00100062 |
| 28 | .04 | .00297508 | .00025499 |

Low/mid probes show a similar drop. Doubling epsilon reduces Q by roughly a
factor of four, consistent with substantial f32 subtraction noise rather than
a converged quadratic response. Input symmetry residuals were around `1e-7`
in global L2, and model outputs remain f32 even though DFT and subtraction use
f64. Q values must therefore not be interpreted as measured physical coupling
strengths until epsilon convergence is established.

## Runtime, memory and compatibility

| Run | Total wall seconds | Base transition bucket | Diagnostics | Additional probes | Sampled peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Baseline | 39.82 | 1.08 | 1.15 | 8.37 | 311.4 |
| Zero sidecar | 30.19 | 1.05 | 1.44 | .99 | 311.5 |
| Transport | 29.67 | 1.12 | 1.33 | 1.12 | 311.1 |
| Diffusion | 30.39 | 1.20 | 1.45 | 1.14 | 311.4 |

Startup/corpus/checkpoint/hash work dominates total time. Base transition includes
host extraction and component accounting; it is not an uninstrumented G benchmark.
Baseline has nine base evaluations (the final response needs one extra) plus
8 paired-growth and 24 ±response evaluations. The other arms each have eight
base and eight paired-growth evaluations. Diagnostic time includes spectral
statistics and recurrence export; it is substantial relative to transition time.
One clean test compilation overlapped baseline startup, so total times are
illustrative measurements, not a controlled speed comparison.

Recurrence retained nine complete f64 states, **18,443,520 bytes = 17.59 MiB**;
the 9x9 matrix uses **648 bytes**. Process RSS includes model, optimizer, corpus,
all states and scratch; sampled VmHWM is not isolated incremental memory overhead.
A long trajectory should use a larger recurrence stride. An FFT would reduce
spectral cost but has deliberately been deferred from this auditable first pass.

Validation passed:

* Eight focused mathematical/operator/CLI tests from the clean checkout.
* Three existing strict checkpoint/model/world persistence tests.
* Eight existing v9 compatibility regressions.
* Strict Clippy for the new binary, formatting of new Rust modules and Git diff checks.
* Real-checkpoint Parseval accounting at every step, finite values, recurrence
  size/symmetry/diagonal, all artifact SHA-256 receipts, and zero-reference policy.
* Exact equality of baseline and zero-sidecar diagnostic rows and recurrence bytes;
  a separate operator unit test checks disabled-branch state values directly.
* Seven pre-existing top-level source checkpoint-directory files remained byte
  identical; occupied output directories were refused without overwriting summaries.

No checkpoint schema, optimizer layout, training defaults, existing CSVs or image
output names changed. `cargo run` retains `titan_image` as its default binary.
The new analyzer is CPU-only; no OpenCL parity or training-gain claim is made.

## Most informative next experiment

Run a **baseline-only epsilon-convergence sweep**, fixed seed/direction and saved
state, high-band perturbations, offsets 0 and 8, epsilons `.08,.16,.32,.64`.
Look for a stable Q range before comparing mechanisms or interpreting coupling:

```sh
./target/release/titan_develop \
  --config analysis/rmsnorm_followup_2026-09-09_matched/parent/config.json \
  --output analysis/development_epsilon_sweep \
  --steps 8 --seed 42 --response-ages 0,8 \
  --response-bands high --response-epsilons .08,.16,.32,.64
```

This next experiment has not been run. Jointly trained transport/diffusion
projections, stress, timescale groups, staged spatial schedules, event-profile
collapse and metastability detection remain separate future work.

## Exact files changed by this first pass

`Cargo.toml`, `README.md`, `src/persistence.rs`,
`src/development/main.rs`, `src/development/metrics.rs`,
`src/development/spectral.rs`, `src/development/operators.rs`,
`scripts/development_smoke.py`, `docs/DEVELOPMENTAL_DYNAMICS.md`,
`docs/DEVELOPMENTAL_VALIDATION.md`.
