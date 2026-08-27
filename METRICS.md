# TITAN Image v9 metrics and experiment protocol

No scalar proves reconstruction, useful emergence, homeostasis, or an
attractor. v9 keeps grounding, emergence, coherence, separability, stability,
and cost visible as distinct measurements.

## Per-window CSV

`titan_image_metrics_v9_<tag>.csv` has one row per optimizer window.

Identity and schedule:

- `step`, `episode`, `age`, `target_index`, `optimizer_update`;
- `objective`, `core_trained`, `macro_updates`, `episode_started`;
- `supervision` (`global` or `crop`), `detail_zoom`, `pyramid_level`;
- `grounding_schedule`, `emergence_schedule`, `reference_fidelity`;
- `active_morph_depth`, `physical_morph_layers`, `morph_generation`.

Grounding:

- `loss_content`: composite raw endpoint error;
- `loss_grounding`: weighted grounded-only multiscale objective;
- `loss_ground_coarse`, `loss_ground_mid`, `loss_ground_fine`;
- `loss_ssim`, `loss_palette`, `loss_structure`; structure is target-aligned
  edge reconstruction in single/family mode and gradient statistics only in
  translation-invariant texture mode;
- `cross_resolution_l1`, `cross_resolution_low`,
  `cross_resolution_edge` on scheduled global consistency windows;
- `loss_endpoint` and `loss_total` remain separate so flow or regularizers
  cannot hide worsening endpoint behavior.

Emergent-head role and contribution:

- `grounded_output_rms`, mean absolute value, and spatial variance;
- `emergent_output_rms`, mean absolute value, and variance;
- `emergent_contribution_rms`: actual composite-minus-grounded image effect;
- `loss_emergent_fit`, `loss_emergent_low`, `loss_emergent_tv`;
- `head_redundancy`: absolute grounded/emergent correlation;
- grounded and emergent gradient RMS, actual update RMS, and update/weight
  ratios.

These are role-collapse diagnostics, not an instruction to maximize residual
energy. A useful residual should increase coherent detail while grounding and
cross-resolution anatomy remain healthy.

State and coherence:

- micro/macro mean and maximum movement;
- state RMS, mean absolute value, near-bound fraction, and channel RMS range;
- interface memory RMS;
- image delta mean/RMS with an explicit validity bit; target, resolution, or
  crop-view transitions invalidate the comparison;
- image variance and edge energy;
- seam energy and gamut excess;
- state/memory barrier losses and the stability-watchdog flag;
- local micro/macro reference-drive RMS.

Optimizer and cost:

- global/core/decoder/grounded/emergent/flow gradient RMS;
- global gradient norm and clip scale;
- effective learning rate, updated tensors/parameters, Muon tensor count;
- window seconds and development steps/second.

Flow windows add:

- exact conditional flow loss and sampled time;
- interpolant, predicted velocity, and target velocity RMS;
- velocity cosine alignment;
- one-step endpoint L1;
- recurrent condition RMS.

## Per-target report

`titan_image_target_statistics_v9_<tag>.json` accumulates per-invocation target
means for total, grounding, content, structure, palette, flow, gradient demand,
clip rate, movement, and memory. Global and detail grounding are split. Mature
episode windows receive separate loss and movement summaries, and grounding
adaptation per window is reported.

Fixed-seed/age family separability analysis adds pairwise:

- output L1;
- low-frequency image distance;
- edge distance;
- micro-state, macro-state, memory, and emergent-residual distances;
- mean/minimum output distance and nearest/confusable target pair.

An attractive shared phenotype with weak target separation should therefore be
immediately visible.

## Model statistics

`titan_image_model_stats_v9_<tag>.json` inventories every parameter tensor:

- subsystem and MorphicBlock identity;
- allocated/active parameter count and inactive reserve;
- birth generation;
- weight RMS, maximum absolute value, and exact-zero fraction;
- row-energy participation ratio for matrices;
- first- and second-optimizer-moment RMS.

Normal windows supply head/subsystem gradient and update telemetry. Full
per-layer activation hooks and expensive singular-value decompositions are not
run continuously on the phone.

## Variable-shape JSONL events

`titan_image_events_v9_<tag>.jsonl` is the append-only developmental record.
Graft events include transaction IDs, old/new anatomy, birth generations,
copied/new tensors and parameters, preserved/new moments, explicit empty
resized/skipped lists, pre-graft losses/state/output fingerprint, and immediate
preservation error. Morph activation events record depth, plateau behavior,
seam, and function-preservation L1.

## Emergence frontier

`titan_image_emergence_frontier_v9_<tag>.json` evaluates one frozen state at
five residual strengths. Each point records content and multiscale grounding,
structure, residual magnitude/low-band/TV/redundancy, seam, gamut, variance,
and edge energy. The corresponding montage uses the same state and seed.

Interpret this as a tradeoff curve. Do not select the point with maximum
residual energy. Prefer the largest coherent elaboration whose coarse identity,
target separation, gamut, and stability remain acceptable.

## Autonomous and perturbation analysis

Autonomous rollout records offset, interval-averaged micro/macro movement and
RMS/near-bound occupancy, memory RMS, image delta, nearest prior-state
signature distance, output fingerprint, and a
conservative approximate-cycle flag. Continued wandering alone is not labeled
a strange attractor. First-sample image/recurrence distances carry explicit
validity flags rather than using a sentinel value.

Perturbation analysis evolves one untouched mature control beside deterministic
micro, macro, and memory noise plus localized micro/macro erased patches. It
reports initial/final state distance, output L1 from control, recovery ratio,
and optional half-recovery time. The JSON uses
`perturbation_recovery_observed=true` only when final state distance is below
half its initial value; phenotype-family recovery can still differ from exact
raster recovery.

## Attribution versus causality

The decomposition montage renders the exact same frozen world as target,
grounded, residual, composite, state-only, learned-only, micro-zero, and
macro-zero variants. These answer where information is currently exposed.

Dynamics ablations clone and continue the world with interface, micro, macro,
NCA, reaction, phase, cyclic, or external forcing disabled/frozen. These answer
which mechanisms affect continued development. Render-time zeroing is not
described as causal proof.

## Deterministic benchmark

`--benchmark` uses seven asymmetric synthetic targets: circle, offset square,
diagonal, checker/grid, nested shapes, asymmetric blobs, and branching form.
It records raw L1, coarse spatial L1, edge L1, development-age convergence, and
candidate separability. The asymmetry specifically prevents palette/statistics
matching from masquerading as target-specific reconstruction.

## Held-out natural-image probe

`--probe-dir` adds a checkpoint-only natural-image transfer matrix. Each JSON
point records target name and source-byte fingerprint, reference fidelity, age,
emergence schedule, output path, registered raw L1/L2, 8x8 coarse spatial L1,
edge L1, palette-mean L1, image variance/edge/seam/RGB means, micro/macro state
RMS and near-bound occupancy, memory RMS, and micro/macro reference-drive RMS.
The report also records both corpus fingerprints, checkpoint world step, fixed
world seed, fixed-zero-genome policy, output counts, and explicit frozen-weight/
zero-optimizer-step flags.

Training and probe manifests are checked for exact source-byte overlap before
any probe trajectory runs. Every target and fidelity receives the same fresh
world seed and checkpoint anatomy. Fidelity zero supplies no reference tensors;
its drive RMS must be exactly zero and same-age outputs must be identical across
targets. Treat that row as a target-independent prior baseline. Positive
fidelities measure transfer to held-out inputs. Neither result alone establishes
broad natural-image generalization beyond the curated probe set.

## Recommended experiment sequence

1. Run `strict-reconstruct` to establish literal global and target-specific
   fidelity.
2. Compare `reconstruction-plus` with identical corpus, seed, steps, and wall
   conditions.
3. Evaluate the emergence frontier and decomposition montage.
4. Run target separability and resolution ladder analysis.
5. Run the held-out natural-image age/fidelity matrix, including fidelity zero.
6. Only then test `grounded-emergent`, adaptive depth, perturbation recovery,
   and autonomous rollout.
7. Test `flow-reconstruct` as a separate matched experiment; do not compare a
   flow sample to the canonical phenotype without labeling it.

Use world-step or wall-clock matched runs and compare peak RSS and measured
development steps/second. Termux background load is part of the observation,
not a constant device property.
