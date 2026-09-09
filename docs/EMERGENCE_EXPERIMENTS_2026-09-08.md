# Controlled emergence experiments — 2026-09-08

This implements a controlled experimental platform around the normalization defect
identified in the historical, locally retained
`analysis/EMERGENCE_AUDIT_2026-09-07.md`. That audit is unchanged.
A correct derivative is a mechanical requirement, not evidence of improved emergence.

## Mechanical change and compatibility

Candle 0.10.2's contiguous `RmsNorm::forward` uses `apply_op2_no_bwd`. This
removes input and scale dependencies through normalized branches. Residual
bypasses and downstream layers still learn. The new
`experiment.norm = "differentiable"` selects `forward_diff` in **tracked**
attention, feedforward, and every active Morphic block. Untracked inference and
the explicit `"legacy"` training baseline retain the prior implementation.
All tensor names, shapes, and numerical bounds are retained.

Old JSON configurations deserialize to the legacy baseline. With no training
experiment enabled, the historical checkpoint signature is unchanged. Norm mode
and withdrawal configuration extend the signature with a versioned domain;
output-only diagnostics and panel controls do not. Ordinary resume still checks
its immutable signature and transaction components. Inference from an old
checkpoint therefore uses its legacy configuration; opting into new training
requires an explicit fork.

Tests cover multiple widths, scales, noncontiguous layouts, forward tolerances,
finite and reachable input/scale gradients, finite differences, all interface
normalization branches, and the CPU-to-OpenCL full-core renderer VJP boundary.
Full-core recurrent differentiation remains on CPU; the OpenCL renderer's input
VJP is composed with it. This is not a claim that RMSNorm runs on the GPU.

## Explicit import

```
target/release/titan_image fork REQUEST.json
target/release/titan_image --config-json DEST/config.json
```

`REQUEST.json` contains `parent_metadata`, a complete `destination` RunConfig,
`optimizer` (`retain` or `reset`), `world` (`retain` or `reseed`), and `warmup`
(`retain` or `reset`, matching optimizer policy). A new output directory and a
new run tag are mandatory. Import creates checkpoints but performs no training.
A `.fork-incomplete` marker prevents resuming a partially published import.

The parent is loaded using its own saved config and the strict component loader.
Import never invokes previous-generation recovery, which can republish files in
the parent directory. No shape resizing or anatomy graft is allowed in this
operation. SHA-256 identities cover parent model, world, optimizer, manifest and
metadata; `fork.json` includes the parent transaction, run identity, optimizer
update count, policies, complete recursive setting differences, build identity,
seed, and requested budget. The destination checkpoint is reloaded under its own
configuration before import completes. Evaluations embed fork provenance.

Retain keeps moment values and the optimizer update count exactly. Its existing
warmup position is retained; it does not restart warmup for newly reachable
normalization parameters. Reset creates zero moments and update count zero, so
configured warmup starts again. Partial optimizer migration and independent
warmup restart are deliberately unsupported in v1. A changed learning rate
requires a fork and is recorded, not recommended as a saturation remedy.

Retain-world imports require the same seed, target schedule, BPTT alignment and
episode policy. Reseed deliberately starts world step, age and episode at zero,
using the destination's deterministic target schedule, while retaining checkpoint
Morphic anatomy. For a BPTT comparison that needs reseeding, reseed **both** its
control and candidate. Comparing a retained-state BPTT-4 run with a newly seeded
BPTT-8 run would confound the comparison.

The first normalization comparison retains optimizer and world state. Names and
shapes match, and preserving accumulated moments avoids introducing a second
large intervention. Previously unreachable normalization scales have zero saved
moments; their first effective updates and clipping need measurement.

## Instrumentation

Analysis-only interface inspection reports separate micro/macro logit RMS,
signed and absolute min/p01/p05/p50/p95/p99/max, fraction above
`atanh(0.99)`, theoretical mean tanh derivative and fractions below 0.01/0.001,
and mean per-channel spatial write variance. Shared token RMS is recorded after
each recurrent loop; GRU reset/update gates have the same distribution summaries.

Sensitivity probes add 0.05 independently to state, references, memory or genome
and report micro/macro write RMS and maximum changes. They are local directional
probes, not full Jacobian norms. Reference sensitivity at fidelity zero should
be exactly zero. Instrumentation is outside normal forward execution and does
not change its output. Large ablation effects do not imply responsive computation.

Optional optimizer JSONL contains group gradient L2, RMS over all group elements,
reachable and nonzero element counts, actual update RMS (including decay),
pre-update parameter RMS, update/parameter ratio, and share of global gradient
energy subject to clipping. Ratios are null for zero parameter energy. Groups
partition NCA scales, interface projections/attention/feedforward, GRU, Morphic
blocks, renderer/shared and heads, reference drives and normalization scales.
Normalization scales are their own group, not double counted in their modules.

Window records include full-core/decoder-only counts, effective BPTT, actual macro
updates, elapsed time and peak RSS. Existing training metadata retains overall
wall time, peak RSS, completed steps and throughput. Loss-component gradient
cosines are deferred: they require additional backward passes and an explicit
sampling/cost policy; current reports do not imply they were measured.

## Reference withdrawal and later temporal experiments

The existing `reconstruct` mode always uses maximum fidelity. Its nominal
`reference_dropout` applies only under `hybrid`; that baseline is retained and
help text now describes the distinction.

An example independent experiment uses:

```json
{
  "norm": "differentiable",
  "withdrawal": {
    "observe": 16,
    "taper": 16,
    "autonomous_tail": 32,
    "min_fidelity": 0.0,
    "max_fidelity": 1.0,
    "guided_anchor_probability": 0.25
  },
  "optimizer_diagnostics": true,
  "panel": null
}
```

This requires a fixed 64-step episode with resetting age. Phase durations are in
developmental steps and sum to episode length. Reference strength is evaluated
on **each step**, including within a BPTT window. Taper interpolates to its minimum;
the tail always has zero fidelity, even when the taper minimum is positive.
A seed/episode hash chooses fully guided anchor episodes. Both pooled interface
references and local RGB drive are absent in autonomous steps. The target image
remains available to the loss. Legacy CSV fidelity is the endpoint value for a
scheduled window; reference-drive RMS columns average over the whole window.

BPTT 8 and `core_update_every=1` are existing supported settings, each signed and
therefore changed through a separate fork. The runner computes a development-step
budget from the saved global step to match an exact count of full-core updates.
Compare matched updates, steps, and measured time separately; those budgets are
not interchangeable. Do not silently combine BPTT and cadence changes.

`src/replay.rs` is a tested scaffold, **not wired into training**. Its constructor
requires an explicit reset-on-resume policy; capacity is bounded to 16 detached
states. Entries carry source fingerprint and fresh/mature/damaged category;
reservoir selection and target/category sampling are seeded. A future training
integration must record this reset policy and sample/insert events. No hidden
pool is active in these experiments and no replay persistence is implied.

## Frozen panel

`experiment.panel` specifies target indices, fixed seeds, real guided burn-in,
diagnostic ages, autonomous horizons (including 128/512/2048), sample stride,
recovery horizon, bounded recurrence-history capacity and a separate optional
clock-sequence robustness case. It is usable with `--analysis-only`.

Each target/seed starts a fresh deterministic world with saved active anatomy,
actually develops through its guided burn-in, then removes both references.
Raw PNGs are saved at every sampled autonomous offset and requested horizon.
All files use the existing write-once evaluation directory and write receipts.
Published provenance now includes SHA-256 alongside legacy FNV/length identities;
experimental frame references are validated against the writer's owned outputs.
The prior saved-state/promoted-age autonomous evaluation remains available under
its original option and is separately identified by provenance.

Recurrence compares full latent fields plus memory by RMS distance and retains
spatial latent correlations. Additional comparisons use 32px RGB Pearson
correlation, 8px low-frequency correlation, edge-map correlation, standardized
brightness-normalized RMS, and a small periodic translation search (+/-2 pixels
on the 32px comparison grid). Constant images return null correlations. Lagged
comparisons include moving states without requiring near-zero movement. The
bounded history defines the searched lag range; these diagnostics do not prove a
fixed point, limit cycle, chaos, strange attractor, or homeostasis.

Recovery separately tests guided and autonomous development after the same true
burn-in. Cases include micro/macro noise, micro/macro central-patch erasure,
memory noise and combined field+memory noise. Noise uses the prior uniform
+/-0.03 perturbation followed by its smooth bound; actual initial distance is
measured. Paired trajectories share world age/step and the same stochastic clock
sequence. The optional `clock_sequence_only` case changes clock selection only,
not physical forcing time or macro cadence.

Reports include state distance, rendered error, target-relative errors for both
trajectories, structural differences, state time-to-half at step resolution,
rendered time-to-half at sampling resolution, and explicit failure to halve.
Short guided recovery is not labelled autonomous repair.

## Objective and saturation experiments remain separate

The preserved emergent-fit forward objective is effectively composite
reconstruction error; stop-gradient on the grounded target residual changes
routing, not the desired visible image. Existing `ObjectiveMode`/`visual_loss`
separation remains the extension point for a later mature objective. Grounded
identity, patch/feature statistics, bounded seed diversity and temporal
organization need separate gates and validation. Neither an entropy reward nor
rectified-flow sampling time establishes organism developmental motion.

There is no write-temperature change, gain reduction, saturation regularizer,
new loss, capacity increase, replay sampling or withdrawal enabled in the initial
normalization A/B. A short pilot cannot establish the retraining horizon needed
to recover saturated writes. Any later pre-tanh penalty must be a separate,
signed opt-in fork and retain numerical bounds.

## Reproduction and pilot evidence

```sh
cargo fmt --check
cargo test --locked --offline
OCL_ICD_ASSUME_ICD_EXTENSION=1 TITAN_OPENCL_TEST=1 \
  cargo test --locked --offline --features opencl
cargo clippy --locked --offline --all-targets --features opencl -- -D warnings
cargo build --locked --offline --release --features opencl

target/release/titan_image gradient-check DEST/config.json

python3 scripts/emergence_experiments.py \
  --parent-metadata /sdcard/Download/titan_image_v9_pure_nca/titan_image_run_metadata_v9_v9-emergent-tuned-c282-01.json \
  --root analysis/NEW_EXPERIMENT_DIRECTORY --core-updates 8 --execute
```

The runner prepares requests without running them unless `--execute` is supplied.
An executed run creates baseline/control/diff forks, frozen CPU gradient probes,
a baseline evaluation, then the two matched continuations and their evaluations.
It verifies every artifact SHA-256 and parent preservation. It refuses existing
experiment directories and does not automatically resume an interrupted pilot.
The initial pilot uses two corpus targets and two fixed world seeds, 64-step
burn-in, 128-step autonomous continuation and 32-step recovery. These are training
corpus probes, not held-out generalization evidence.

Pilot results and final validation are recorded below after completion.
