# TITAN Image: synthesis-focused implementation plan

Goal: generate varied, coherent structure from genome and initial state, with
no runtime reference tensors. Stability and novelty must be evaluated together.

## 1. Add a frozen synthesis panel

- Extend the existing frozen loader/renderer with explicit genome inputs:
  native corpus genomes, controlled mixtures, and bounded mutations that respect
  the existing genome representation. Record the actual vectors and their origins.
- Use matched fresh initial states and clocks for each seed across genome choices;
  retain the checkpoint's anatomy. Keep weights frozen and reference fidelity zero.
- Start with two native genomes, their midpoint, and one small mutation, using two
  seeds. Run serially to 128 steps and save frames at offsets 0, 32, 64, and 128.
  Extend only useful cases to 512 steps.
- Export immutable JSON/CSV receipts, raw renders, and a labelled contact sheet.
  Load the checkpoint once per panel where practical to avoid repeated startup.

## 2. Add per-grid diffusion controls

- In `src/development/operators.rs`, support micro-only, macro-only, and both-grid
  diffusion. Preserve current both-grid defaults, zero-operator behavior, transport
  behavior, and macro update cadence. This remains a frozen experimental sidecar.
- Wire selection into the recovery/synthesis CLI, manifests, and runner. Compare
  off, macro-only, micro-only, and both at fixed effective nu=.01 on a small subset
  of the synthesis panel; avoid multiplying the entire panel immediately.
- Reuse the existing field/frequency residual and image-detail measurements.
  Test excluded-grid preservation, default parity, coefficient validation, and
  spectral/distance closure. CPU remains the numerical oracle.

## 3. Judge synthesis directly

- Measure genome responsiveness, between-output diversity, image variance/edges,
  and persistence or coherent change across developmental ages.
- Use same-genome repeats and seed changes to distinguish conditioning effects
  from stochastic variation. Reject flat outputs and noise as novelty successes.
- Compare generated outputs with the corpus only after generation, using aligned
  and multiscale/edge comparisons. Distance from training images alone is not proof
  of meaningful novelty. Inspect the rendered gallery before claiming quality.
- Prefer a diffusion setting only if its robustness benefit retains useful detail
  and controllable variation. Defer learned diffusion until this gate is met.

## 4. Then test training on the exploratory fork

- Use the authorized fork at `/sdcard/Download/titan_image_v9_dynamics_diffusion`.
  Recheck its current checkpoint and active process before starting; the verified
  snapshot was step 130000, age 16, target 38. Preserve its long-run parent.
- Run a short reference-withdrawal comparison using existing withdrawal support,
  with a declared full-core update budget and before/after synthesis panels.
  Keep other training changes separate; do not restart the full 35000-step budget
  merely to test the idea. Evaluate guided held-out reconstruction separately.
- Any diffusion added to training needs its own opt-in implementation, CPU gradient
  checks, and OpenCL parity. The current frozen sidecar is not a training feature.

## Delivery and working notes

Preserve unrelated `src/run_lease.rs` edits and `fork_diffusion_request.json`.
Use new output roots; retain checkpoint, source, binary, genome, and command
receipts. Validate focused changes, commit/push the implementation, and rebuild
release binaries before empirical runs. Report results and the next decision;
do not equate contraction, smoothing, or corpus distance with successful synthesis.

Prior evidence: [residual decomposition](docs/RECOVERY_RESIDUAL_RESULTS_2026-09-13.md)
and [diffusion/detail tradeoff](docs/FROZEN_DIFFUSION_RECOVERY_RESULTS_2026-09-13.md).
