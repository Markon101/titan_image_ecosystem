# TITAN Image Ecosystem v9

TITAN Image 0.9.1 is a phone-first compact morphogenic learner centered on
Reconstruction++: reconstruct the source's grounded organization, then permit
a recurrent developmental organism to add bounded, coherent elaboration
without surrendering source identity.

The intended operating region is not maximum novelty and not minimum pixel
loss in isolation. It is maximum useful emergent organization subject to
grounding, coherence, numerical stability, recoverability, and mobile cost.

v9 is a schema boundary. It does not load v8 checkpoints and writes only v9
artifacts. Within v9, append-only MorphicStack growth is explicitly supported.
The default output root is `/sdcard/Download/titan_image_v9`.

## Build and start

~~~sh
cargo build --release --locked
~~~

Recommended S25-balanced Reconstruction++ run:

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced \
  --research-preset reconstruction-plus \
  --mode family \
  --conditioning reconstruct \
  --objective reconstruction-plus \
  --threads 7 \
  --steps 1600 \
  --run-tag reconstruction-plus-v9-01
~~~

Continue by repeating the learned/evolution settings and tag without
`--fresh`. Output-only and analysis controls may change. A changed physical
`--morph-layers` is accepted only as a validated contiguous append graft.

## Reconstruction++

The renderer exposes three distinct quantities:

~~~text
grounded phenotype = recurrent state -> shared renderer -> grounded head
emergent residual  = recurrent state -> shared renderer -> bounded residual head
final phenotype    = grounded + emergence_strength * scale_shaped(residual)
~~~

Composition occurs in the bounded Oklab-like rendering representation. The
grounded image and emergent residual visualization are saved separately. At
`--emergence-strength 0`, the residual has exactly zero influence on the final
image even if the residual head itself is nonzero.

The grounded path is trained with fine, medium, and coarse reconstruction,
SSIM-like structure, palette, target-aligned spatial edges, gamut, state, and memory
terms. Coarse and medium scales receive the strongest default grounding. The
emergent head is zero-initialized, magnitude-bounded, monitored separately,
regularized against redundant head behavior, and shaped by low/mid/high
frequency budgets. Fine-scale freedom is the initial bias, not a permanent
prison: `grounded-emergent` and `free-morph` permit increasing mesoscale and
limited low-frequency access.

The developmental schedule is smooth:

- early age establishes the body plan with full grounding and zero emergence;
- middle age ramps morphology and residual contribution;
- mature age retains a nonzero grounding floor while using the configured
  emergence strength.

The resolved grounding and emergence schedule values are logged every window.

### Variable developmental ages

The legacy default remains one fixed `--episode-steps` horizon and preserves
existing checkpoint signatures and trajectories. Once an age range is supplied,
`--episode-steps` no longer controls boundaries or checkpoint identity. New runs
sample a deterministic, BPTT-aligned horizon for each corpus episode:

```sh
# Immediate uniform sampling from 32,36,...,96 steps.
--age-min 32 --age-max 96 --bptt 4

# Expanding curriculum: begin at 32, then grow the eligible upper bound to 96
# over 20,000 global development steps; continue sampling the full range after.
--age-min 32 --age-max 96 --age-curriculum-steps 20000 --bptt 4
```

Both age bounds are required together, must be multiples of `--bptt`, and are
limited to 4096. Sampling is a pure function of the run seed, episode index, and
episode start step, so checkpoint resume needs no new world tensor or schema
change. The developmental conditioning/schedules use `--age-max` as their
absolute mature-age scale; a shorter episode therefore supervises an earlier
point on the same developmental clock rather than compressing maturity into
fewer steps.

Uniform sampling is the recommended first experiment. Growing NCA work uses
random rollout counts to train persistence across an interval, and TITAN already
applies a loss at every BPTT window while traversing each sampled episode. The
expanding curriculum is opt-in for runs that are unstable at long horizons; it
keeps sampling shorter horizons after longer ones become eligible, avoiding a
hard phase switch and reducing early-age forgetting. See
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) and the
original [curriculum-learning paper](https://icml.cc/2009/papers/119.pdf).

## Research presets

Every preset is applied before explicit scalar overrides, regardless of CLI
argument order. All resolved values are persisted in run metadata.

| Preset | Purpose |
|---|---|
| `strict-reconstruct` | Maximum literal grounding and nearly disabled emergence |
| `reconstruction-plus` | Stable balanced default |
| `grounded-emergent` | More late and mesoscale freedom with low-frequency identity protection |
| `free-morph` | Weak/optional reference and synthesis-oriented development |
| `flow-reconstruct` | Experimental endpoint plus exact conditional rectified flow |

Strict reconstruction:

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced \
  --research-preset strict-reconstruct \
  --mode family \
  --threads 7 \
  --steps 1600 \
  --run-tag strict-reconstruct-v9-01
~~~

Higher-emergence Reconstruction++:

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced \
  --research-preset grounded-emergent \
  --emergence-strength 0.80 \
  --mode family \
  --threads 7 \
  --steps 1600 \
  --run-tag grounded-emergent-v9-01
~~~

## Native-resolution supervision

v9 separates five resolutions:

1. original source resolution;
2. 64x64/32x32 developmental field resolution in the balanced profile;
3. 192px global supervision resolution;
4. 128px zoomed local-detail supervision drawn from higher source levels;
5. 768px default final output resolution.

Sources are preflighted at startup and receive a persistent, fingerprinted
Lanczos pyramid under `pyramid_cache_v9` (or `--pyramid-cache-dir`). Global
views teach composition and low-frequency organization. Deterministic crops
sample a log-uniform zoom curriculum and preserve normalized phenotype
coordinates. The renderer sees those global coordinates plus a bounded LOD
signal, so a crop is a zoomed observation of the same organism rather than an
unrelated target.

The current aspect policy is deliberate center-square crop, never stretching.
Original width, height, aspect ratio, crop transform, and available pyramid
levels are stored in metadata. This preserves circle geometry but does discard
the outer sides of non-square sources; padded/canvas supervision remains a
future extension.

Natural images, local crops, and periodic textures use distinct target-boundary
loss policies. The recurrent world remains toroidal; natural/crop supervision
does not receive an artificial seam penalty.

Important controls:

~~~text
--detail-crop-probability X
--detail-resolution N
--detail-min-zoom X
--detail-max-zoom X
--detail-curriculum-start X
--pyramid-cache-max-level N
--pyramid-cache-dir PATH
--target-boundary natural|periodic|crop
--loss-cross-resolution X
~~~

The direct local reference projection is a learned bias-free 1x1 RGB-to-state
drive into both recurrent fields. It is bounded, multiplied by reference
fidelity, exactly absent at fidelity zero, and never passed to the renderer.
The only legal copy route is `reference -> organism -> state -> renderer`.

## Architecture

The balanced v9 organism uses:

- 24-channel 64x64 micro and 32x32 macro recurrent fields;
- local multiring NCA plus optional reaction-diffusion, complex phase, cyclic
  chemistry, quasiperiodic, and stable IFS operators;
- bias-free local RGB reference drives;
- an 8x8 recurrent interface with hierarchical capped global interaction for
  larger grids, including supported 16x16 configurations;
- width-128 GRU/interface memory and three shared reasoning loops;
- six allocated MorphicBlocks with three initially active;
- a shared implicit renderer with separate grounded and bounded emergent heads;
- an always-allocated compact experimental flow head, inactive unless selected.

The renderer never accepts raw reference pixels. The recurrent organism remains
the primary phenotype store, with a parameter-free state-skip making physical
state visibly auditable.

## Recurrent stability

The v9 defaults incorporate the collapse postmortem directly:

- `dt=0.12`, `nca_gain=0.25`, and `state_leak=0.10`;
- one full-core window for every decoder-only window (`core_update_every=2`);
- smooth quartic state projection rather than hard clamping;
- bounded local reference and interface writeback;
- `memory_limit=3.0`, `morph_residual_gain=0.02`;
- state and memory soft barriers;
- global gradient clipping at 1.0 and 48-update warmup;
- finite checks on recurrent movement, reference drives, flow tensors, flow
  loss, ODE states, optimizer gradients, Muon directions, and loaded moments;
- a near-bound watchdog that saves a normal resumable checkpoint instead of
  continuing a numerical collapse.

This allows richer morphology without treating limit collisions, exploding
gradients, or unbounded direction vectors as useful emergence. Psychedelic
chaos can still be a fun ablation; it is not the default optimizer objective. 🙂

## Morphic growth and v9 grafts

`--morph-layers` is allocated physical capacity. Active depth lives in the
world checkpoint and is passed into the interface on every step.

- `fixed`: use `--morph-depth`;
- `capacity`: activate the configured maximum reserve;
- `adaptive`: begin at `--morph-min-depth` and conservatively activate reserved
  blocks after a stable grounding plateau.

New blocks have zero contract/writeback parameters. The stack performs one
post-stack smooth projection, so activating a reserved zero-contract block is
exactly function-preserving.

Append-only physical grafting preserves all old tensors and optimizer moments,
initializes only the exact five tensors per new MorphicBlock, rejects unknown,
renamed, resized, malformed, missing, or noncontiguous tensors, and records
parameter birth generation. A transaction ID must agree across model, world,
optimizer, and manifest; the manifest is published last. The previous complete
generation is retained and automatically restored if the current transaction
is incomplete or corrupt.

Adaptive example:

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced \
  --research-preset reconstruction-plus \
  --morph-depth-mode adaptive \
  --morph-layers 8 \
  --morph-min-depth 2 \
  --morph-max-depth 8 \
  --morph-growth-interval 512 \
  --threads 7 \
  --steps 3200 \
  --run-tag adaptive-morph-v9-01
~~~

## Optional conditional rectified flow

TITAN's NCA trajectory is not relabeled as flow matching. v9's experimental
flow head learns a separate low-resolution Oklab velocity field conditioned on
frozen recurrent micro/macro observations and interface memory.

For target endpoint `x1`, deterministic Gaussian `epsilon`, and sampled
`t ~ U(0,1)`:

~~~text
x_t = (1 - t) epsilon + t x1
v*  = x1 - epsilon
L_CFM = mean((v_theta(x_t, t, recurrent_condition) - v*)^2)
~~~

The target transform is fixed, not learned. The velocity API has no reference,
target, genome, grounded-head, or emergent-head input. Hybrid flow uses
`L = endpoint_weight * L_Reconstruction++ + flow_weight * L_CFM`; the preset
starts conservatively at flow weight 0.10. Midpoint ODE samples are separate
analysis artifacts and never replace the canonical composite phenotype. No
likelihood claim is made because divergence is not computed.

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced \
  --research-preset flow-reconstruct \
  --mode family \
  --threads 7 \
  --steps 1600 \
  --run-tag flow-reconstruct-v9-01
~~~

Flow remains global and 64px by default. Native-resolution crop flow is
intentionally deferred because independent crop noise would violate the
same-anatomy-across-scale contract.

## Analysis and scientific interpretation

Analysis clones and detaches the saved world. It does not mutate optimizer
state or the training world.

Checkpoint analysis and render attribution:

~~~sh
./target/release/titan_image \
  --analysis-only \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced \
  --research-preset reconstruction-plus \
  --render-attribution \
  --model-stats \
  --run-tag reconstruction-plus-v9-01
~~~

Held-out natural-image transfer and reference-free probe:

~~~sh
./target/release/titan_image \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --probe-dir /sdcard/Download/titan_image_unseen_probes \
  --output-dir /sdcard/Download/titan_image_v9 \
  --run-tag v9-grounded-emergent-c81-01 \
  --profile s25-balanced \
  --style alien-fluid \
  --research-preset grounded-emergent \
  --seed 42 \
  --threads 7 \
  --analysis-only \
  --probe-ages 1,8,16,32,64 \
  --probe-reference-fidelities 1.0,0.5,0.25,0.1,0.0
~~~

`--corpus-dir` must still name the exact training corpus so the saved checkpoint
can pass its normal corpus fingerprint validation. `--probe-dir` is loaded only
after that checkpoint succeeds; it never joins the training schedule, never
changes the checkpoint signature or training-corpus fingerprint, and is
rejected if any source bytes match the training corpus. Probe mode requires
`--analysis-only` and a complete checkpoint. It takes no optimizer steps and
preserves model, optimizer, world, manifest, training CSV, and training metadata
bytes.

Every image/fidelity trajectory starts from the same fresh deterministic world,
retaining only the checkpoint anatomy, and uses a fixed zero genome. Positive
fidelities therefore isolate transfer through the trained reference pathway. At
fidelity `0`, no reference tensors or local reference drive are supplied; all
targets have the same output at a given age by construction. That row measures
the organism's reference-free developmental prior, not reconstruction of an
unseen target it has no information about.

The sweep writes one PNG per target/fidelity/age, a contact sheet,
`titan_image_probe_report_v9_<tag>.json`, and the same structured report inside
`titan_image_analysis_v9_<tag>.json`. Raw/coarse/edge/palette reconstruction,
image/state/memory stability, and reference-drive metrics are recorded for each
point. Probe ages must be sorted unique values in `1..=4096`; fidelities must be
unique values in `0..=1`. Defaults are the two lists shown above.

Autonomous mature rollout removes both reference tensors and sets reference
fidelity to zero at every continuation step. It retains the saved organism,
its memory, and its fixed genome; this tests reference withdrawal from a
developed state, not the fresh zero-reference prior. Analysis version 2 records
zero reference-drive RMS explicitly. Older rollout reports used ongoing reference
guidance and must not be compared as the same experiment.

Autonomous mature rollout:

~~~sh
./target/release/titan_image --analysis-only \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced --research-preset reconstruction-plus \
  --autonomous-rollout 512 --analysis-stride 32 \
  --run-tag reconstruction-plus-v9-01
~~~

Perturbation recovery and causal dynamics ablations remain reference-guided;
recovery records explicitly include the configured fidelity. Legacy case names
ending in `_gaussian` are retained for readers, with an added field identifying
the actual deterministic uniform noise distribution.

Completed analysis summaries now include additive provenance: analysis version,
evaluation ID, build/configuration, initial state fingerprints, on-disk checkpoint
identities, diagnostic completion flags, and fingerprints of generated artifacts.
Each complete JSON summary is also retained in `analysis_history_v9_<tag>/`.
Existing latest-summary and sidecar filenames remain available. Sidecars absent
from the completion/artifact inventory belong to another evaluation or were not
requested. Image paths remain mutable; verify the recorded fingerprint before
using an image with an archived report. This adds no checkpoint/schema migration
and does not alter training trajectories or the CSV format.

Perturbation recovery and causal dynamics ablations:

~~~sh
./target/release/titan_image --analysis-only \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --profile s25-balanced --research-preset reconstruction-plus \
  --perturbation-analysis 256 --dynamics-ablation 128 \
  --run-tag reconstruction-plus-v9-01
~~~

The deterministic asymmetric reconstruction benchmark is enabled with
`--benchmark`. Family analysis renders up to eight targets from a fixed seed
and age and reports output, low-frequency, edge, micro-state, macro-state,
memory, and emergent-residual separation. `--compare-v8-dir PATH` compares
compatible saved v8 metrics/metadata with the current v9 run; it does not load
v8 checkpoints.

Reports use conservative language. Continued motion is not automatically a
strange attractor; damage recovery is not called homeostasis unless measured;
render-time zeroing is attribution, not proof of dynamical causality; and v9
does not claim metaphysical strong emergence.

## Terminal and telemetry

`--terminal compact|rich|quiet` controls training output. ANSI is used only for
a real TTY. Compact and rich lines include development steps/second; rich mode
adds the grounding/emergence/movement/stability tape. Crop windows are marked
`CROP Nx`, global windows `GLOBAL`.

Temporal image deltas are valid only when consecutive renders use the same
target, resolution, and observation. Global/crop or crop/crop view changes are
logged with `image_delta_valid=false` instead of comparing unrelated rasters.

v9 leaves a developmental medical record:

- `titan_image_metrics_v9_<tag>.csv`: flat per-window telemetry;
- `titan_image_events_v9_<tag>.jsonl`: graft/activation/development events;
- `titan_image_target_statistics_v9_<tag>.json`: per-target rolling summaries;
- `titan_image_model_stats_v9_<tag>.json`: parameter and optimizer-moment stats;
- `titan_image_analysis_v9_<tag>.json`: combined checkpoint analysis;
- separate graft, attractor, perturbation, frontier, benchmark, comparison, and
  flow artifacts;
- raw, mastered, grounded, emergent, decomposition, target comparison,
  resolution ladder, state atlas, preview, and gallery PNGs.

See [METRICS.md](METRICS.md), [RESEARCH_V9.md](RESEARCH_V9.md), and
[math.md](math.md) for metric definitions, design boundaries, and equations.

## Phone boundary

The S25-balanced path keeps the 64x64/32x32 world compact, uses 192px global
supervision and occasional 128px crops, caches source pyramids on NAND, and
moves heavy SVD-like, rollout, decomposition, separability, and high-resolution
work out of normal optimizer windows. The 8x8 interface is the balanced
reconstruction default; 16x16 uses a capped hierarchical global interaction
instead of unrestricted quadratic attention.

The flow head is always allocated for v9 checkpoint stability but inactive
unless selected. Its ODE sampling is analysis-only. Actual RAM, speed, and
phase timing are reported per invocation; README numbers are not promises for
every Termux background load.

## Verification

~~~sh
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo build --release --locked
~~~

The tests cover schema/config semantics, deterministic variable-age scheduling,
window-shared reference-drive parity, decomposition isolation, native-detail
coordinates, cross-resolution consistency, 16x16 hierarchical interface,
long recurrent boundedness, exact flow math, frozen ODE non-mutation,
append-only model/optimizer migration, transaction integrity, checkpoint
resume, CSV alignment, and Ctrl-C-safe artifact completion.
