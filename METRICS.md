# TITAN Image v6 metrics and experiment protocol

TITAN Image deliberately reports several independent views of training. No scalar establishes visual quality, novelty, learning, fractality, or a useful attractor by itself. Inspect the raw image, state atlases, trajectory metrics, corpus target, and matched controls together.

## Sampling geometry

v6 writes one CSV row per optimizer window, not one row per development step. A window contains `bptt` recurrent steps followed by one endpoint render, loss, backward pass, and AdamW update. v6 extends the v5 row with scale-separated state motion, near-bound occupancy, image-space motion, and parameter-count-normalized gradients.

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
- `episode_started`: whether the organism was freshly seeded or blended immediately before this window.

### Dynamical state

- `micro_movement_mean` / `micro_movement_max`: mean and largest per-step mean absolute micro-state change in the window.
- `macro_movement_mean` / `macro_movement_max`: mean and largest slow-field change over actual macro updates; zero when no macro update occurred.
- `micro_state_rms` / `macro_state_rms`: endpoint RMS for each spatial scale.
- `micro_state_mean_abs` / `macro_state_mean_abs`: endpoint mean absolute amplitude for each scale.
- `micro_clamp_fraction` / `macro_clamp_fraction`: fraction of values at or above 99% of the configured symmetric state bound.
- `*_channel_rms_min` / `*_channel_rms_max`: smallest and largest per-channel RMS at each scale.

Movement must be interpreted with state scale, near-bound occupancy, channel range, and images. Near-zero movement can be convergence, collapse, or a renderer-only solution. High movement can be sustained morphogenesis or unstable noise. A rising RMS is not clamp reliance unless near-bound occupancy rises with it.

### Rendered image

- `image_variance`: variance over all RGB values.
- `seam_energy`: mean squared discrepancy between opposite first/last rows and columns.
- `edge_energy`: mean squared horizontal/vertical finite difference.
- `gamut_excess`: mean amount by which pre-clipped linear RGB lies outside `[0,1]`.
- `red_mean`, `green_mean`, `blue_mean`: per-channel means.
- `red_variance`, `green_variance`, `blue_variance`: per-channel variances.
- `rg_correlation`, `rb_correlation`, `gb_correlation`: centered pairwise RGB correlations.
- `image_delta_valid`: false only when no prior optimizer-window render is available, such as the first row of an invocation.
- `image_delta_mean` / `image_delta_rms`: mean absolute and RMS RGB change from the preceding optimizer-window render. Treat episode-start deltas separately because the world and target may both change.

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
- `gradient_rms`: `gradient_norm / sqrt(updated_parameters)`, making scale easier to compare after width changes.
- `gradient_clip_scale`: multiplier applied to every gradient; `1` means no clipping.
- `updated_variables` / `updated_parameters`: tensors and scalar parameters reached by this window's loss graph.
- `effective_learning_rate`: peak learning rate times persisted warmup gain.
- `window_seconds`: wall time from the beginning of recurrent development through metric construction for this optimizer window. Snapshot and checkpoint time are excluded.

Repeated clip scales far below one indicate either an aggressive clip threshold or unstable gradients. A high loss with very small gradient can indicate saturation or an objective with weak sensitivity. Warmup is keyed to the persisted optimizer update count, so continuation does not restart it.

The v6 default global clip is 1.5 rather than v5's 1.0. Global L2 norm tends to scale with the square root of parameter count, so this preserves approximately the same per-parameter clipping pressure for the 2.06x balanced network. Use `gradient_rms` for cross-width comparisons.

## Run metadata

`titan_image_run_metadata_v6*.json` records:

- package/schema version and build commit/dirty/release status;
- exact invocation and complete resolved configuration;
- requested configuration plus effective thread count;
- source count, resized cache count, and content-based corpus fingerprint;
- configuration signature and parameter count;
- requested versus completed development steps and truthful interrupted status;
- start/end world steps and AdamW update counts;
- resume status and restored optimizer-tensor count;
- requested/completed optimizer windows and full-core versus decoder-only completed counts;
- peak resident set size from Linux `VmHWM` when available;
- every output path;
- the index/name/content-fingerprint mapping for every corpus source;
- requested/completed gallery counts and compile-time Rust flags;
- phase totals for corpus startup, tracked dynamics, detached dynamics, render/loss, backward, optimizer, metrics/logging, output rendering, checkpointing, and total wall time.

`average_ms_per_development_step` includes startup and final output work. For steady training throughput, prefer the CSV `window_seconds / bptt` after early warmup. For end-to-end phone budgeting, use the metadata total.

Android may change CPU affinity or frequency during a run. Metadata's effective thread count reveals affinity restrictions but cannot by itself report temperature or clock throttling. Record surface/battery temperature, charging state, screen state, and foreground/background state beside thermal experiments.

Render-only invocations write `titan_image_render_metadata_v6*.json` so exploration does not overwrite the training record or mutate checkpoints.

## Visual evidence

- `raw`: direct differentiable renderer output; use this for architecture comparisons.
- `mastered`: toroidal local contrast, bloom, and shoulder; use this for presentation.
- `micro_state` / `macro_state`: all recurrent channels tiled with independent signed normalization. Cyan and magenta are opposite signs; tile colors cannot be compared as absolute amplitudes.
- `snapshot`: lower-cost periodic preview. Its filename records global step, episode, organism age, and target index. The nominal cadence is rounded up to a safe optimizer-window boundary when it is not divisible by BPTT.
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
- per-channel state entropy and cross-channel effective rank;
- multiscale radial power-spectrum slope over a declared fit range;
- structural similarity and optical-flow statistics between developmental frames;
- regeneration curves after controlled damage;
- diversity distance across genome interpolation, seeds, and ages;
- a scale-range sensitivity plot for any box-counting estimate;
- held-out source-family evaluation when the corpus is large enough.

Do not label a box-counting estimate “the fractal dimension” without resolution and fit-range stability, and do not call an in-corpus fallback a held-out validation set.
