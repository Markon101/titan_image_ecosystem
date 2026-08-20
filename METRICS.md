# TITAN Image v5 metrics and experiment protocol

TITAN Image deliberately reports several independent views of training. No scalar establishes visual quality, novelty, learning, fractality, or a useful attractor by itself. Inspect the raw image, state atlases, trajectory metrics, corpus target, and matched controls together.

## Sampling geometry

v5 writes one CSV row per optimizer window, not one row per development step. A window contains `bptt` recurrent steps followed by one endpoint render, loss, backward pass, and AdamW update. This is a breaking semantic change from v4.

`core_trained=true` means gradients traversed the recurrent trajectory and updated any reached NCA parameters. `core_trained=false` means the NCA weights were detached during forward development and only the renderer received gradients. Physical operators have no learned parameters, but their state paths can still carry a gradient during a full-core window.

## CSV fields

### Identity and schedule

- `step`: global completed development steps. It never resets between episodes.
- `age`: development age of the current seeded organism. It resets when `episode_reset > 0` and a new source episode begins.
- `episode`: coherent target episode number.
- `target_index`: index in content-fingerprint order, not filename order.
- `optimizer_update`: persisted AdamW update count after this row's update.
- `core_trained`: whether this was a full recurrent tape.
- `macro_updates`: number of macro-field updates within this BPTT window.

### Dynamical state

- `movement_mean`: mean over the per-step mean absolute micro-state change in the window.
- `movement_max`: largest per-step mean absolute micro-state change in the window.
- `state_rms`: RMS of the endpoint micro state.

Movement must be interpreted with state scale and images. Near-zero movement can be convergence, collapse, or a renderer-only solution. High movement can be sustained morphogenesis or unstable noise. A rising state RMS near `state_limit` can indicate clamp reliance even if image loss falls.

### Rendered image

- `image_variance`: variance over all RGB values.
- `seam_energy`: mean squared discrepancy between opposite first/last rows and columns.
- `edge_energy`: mean squared horizontal/vertical finite difference.
- `gamut_excess`: mean amount by which pre-clipped linear RGB lies outside `[0,1]`.
- `red_mean`, `green_mean`, `blue_mean`: per-channel means.
- `red_variance`, `green_variance`, `blue_variance`: per-channel variances.
- `rg_correlation`, `rb_correlation`, `gb_correlation`: centered pairwise RGB correlations.

The separate channel statistics close a common loophole: one high global variance can hide a dead or saturated color channel. Correlations distinguish genuine multi-axis color variation from three scaled copies of the same scalar field. This mirrors TITAN Audio's use of width, correlation, and channel balance together rather than trusting a single side-energy value.

Low seam energy supports border compatibility but does not prove low-frequency smoothness. Low edge energy may mean attractive smooth flow or bland blur. High gamut excess means the learned renderer is relying on clipping; the explicit gamut loss preserves a corrective gradient outside the valid range.

### Objective

- `loss_total`: weighted sum used for backward.
- `loss_content`: spatial pixel L1 in `single`/`family`, or normalized RGB correlation plus normalized multi-lag autocorrelation in `texture`.
- `loss_palette`: channel mean error plus log-contrast error.
- `loss_structure`: log mean-absolute-gradient and log gradient-RMS mismatch, separately by channel and axis.
- `loss_seam`: the unweighted seam term.
- `loss_gamut`: the unweighted gamut-excess term.

The configured weights are recorded in run metadata. Compare unweighted components as well as total loss. Normalized texture terms are scale-aware but still do not measure semantics or human aesthetic preference.

### Optimizer and throughput

- `gradient_norm`: global L2 norm before clipping.
- `gradient_clip_scale`: multiplier applied to every gradient; `1` means no clipping.
- `effective_learning_rate`: peak learning rate times persisted warmup gain.
- `window_seconds`: wall time from the beginning of recurrent development through metric construction for this optimizer window. Snapshot and checkpoint time are excluded.

Repeated clip scales far below one indicate either an aggressive clip threshold or unstable gradients. A high loss with very small gradient can indicate saturation or an objective with weak sensitivity. Warmup is keyed to the persisted optimizer update count, so continuation does not restart it.

## Run metadata

`titan_image_run_metadata_v5*.json` records:

- package/schema version and build commit/dirty/release status;
- exact invocation and complete resolved configuration;
- requested configuration plus effective thread count;
- source count, resized cache count, and content-based corpus fingerprint;
- configuration signature and parameter count;
- start/end world steps and AdamW update counts;
- resume status and restored optimizer-tensor count;
- full-core versus decoder-only window counts;
- peak resident set size from Linux `VmHWM` when available;
- every output path;
- phase totals for corpus startup, tracked dynamics, detached dynamics, render/loss, backward, optimizer, metrics/logging, output rendering, checkpointing, and total wall time.

`average_ms_per_development_step` includes startup and final output work. For steady training throughput, prefer the CSV `window_seconds / bptt` after early warmup. For end-to-end phone budgeting, use the metadata total.

Android may change CPU affinity or frequency during a run. Metadata's effective thread count reveals affinity restrictions but cannot by itself report temperature or clock throttling. Record surface/battery temperature, charging state, screen state, and foreground/background state beside thermal experiments.

Render-only invocations write `titan_image_render_metadata_v5*.json` so exploration does not overwrite the training record or mutate checkpoints.

## Visual evidence

- `raw`: direct differentiable renderer output; use this for architecture comparisons.
- `mastered`: toroidal local contrast, bloom, and shoulder; use this for presentation.
- `micro_state` / `macro_state`: all recurrent channels tiled with independent signed normalization. Cyan and magenta are opposite signs; tile colors cannot be compared as absolute amplitudes.
- `variant_*_raw`: fresh deterministic organism under an interpolated/mutated genome.
- `variant_*_mastered`: mastered counterpart.
- `gallery`: contact sheet of mastered variants at increasing developmental ages.

A visually rich state atlas with a bland raw image diagnoses a renderer/objective bottleneck. A vivid raw image with near-static or saturated state can diagnose a coordinate or color shortcut. Rich mastered output with weak raw structure diagnoses postprocessing dependence.

## Minimum ablation protocol

For a claim that an operator helps:

1. freeze corpus bytes, mode, profile, explicit overrides, seed, step count, and thread/foreground policy;
2. use a distinct run tag and `--fresh` for baseline and ablation;
3. change exactly one operator gain or `--no-*` switch;
4. repeat at least three seeds;
5. compare raw outputs, both state atlases, and matched-step metric trajectories;
6. report median and dispersion, not only the best seed or gallery frame;
7. inspect gradient clipping and core cadence before attributing a difference to dynamics;
8. record temperature/cpuset conditions for timing comparisons.

The first useful matrix is:

- selected style baseline;
- `--no-reaction-diffusion`;
- `--no-complex-phase`;
- `--no-fractal`;
- `--no-quasiperiodic`;
- `--no-cyclic`;
- `--style pure-nca` as the combined explicit-operator control.

Style presets also change the direct state-to-color basis, so a style-to-style comparison is not an operator ablation. For a causal operator test, keep the same `--style` and change only a gain or `--no-*` flag.

## Attractor and novelty discipline

Treat an attractor as a hypothesis. Evidence should include:

- repeated trajectories from perturbed states or multiple seeds;
- state-space recurrence or contraction estimates, not only a settling movement scalar;
- regeneration after localized damage;
- sustained boundedness without clamp dominance;
- raw image and latent-state behavior over time;
- matched controls that remove the candidate mechanism.

Likewise, genome variants or changing images do not by themselves prove novelty. A useful future novelty analysis would compare normalized multiscale features against both the source corpus and the generator's recent outputs.

## Metrics still worth adding

- time-resolved CPU affinity/frequency and thermal readings where Android permits them;
- state clamp fraction and per-channel state entropy;
- multiscale radial power-spectrum slope over a declared fit range;
- structural similarity and optical-flow statistics between developmental frames;
- regeneration curves after controlled damage;
- diversity distance across genome interpolation, seeds, and ages;
- a scale-range sensitivity plot for any box-counting estimate;
- held-out source-family evaluation when the corpus is large enough.

Do not label a box-counting estimate “the fractal dimension” without resolution and fit-range stability, and do not call an in-corpus fallback a held-out validation set.
