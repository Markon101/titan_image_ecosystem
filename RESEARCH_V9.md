# TITAN Image v9 research record

## Why v9 exists

The v8 raw-reconstruction experiment demonstrated stable recurrent training but
also a decisive failure mode: shared palette and structural statistics could
improve while target-specific spatial identity remained weak. Attractive
common morphology is not reconstruction. v9 therefore treats target grounding,
controlled elaboration, scale consistency, separability, and recoverability as
separate measured properties.

The design target is a compact recurrent organism whose developmental state is
far smaller than the images it can reconstruct and elaborate.

## Reconstruction++ contract

Let `S_mu`, `S_M`, and `m` be micro state, macro state, and interface memory.
The shared implicit renderer produces a grounded Oklab-like field `g` and a
bounded emergent field `e`. The visible phenotype is:

~~~text
y(alpha) = RGB(g + alpha * B_scale(e))
~~~

`B_scale` decomposes the residual into low, medium, and high bands. High bands
have full default access, medium access is configurable, and low access is
small by default but can be opened by research presets. This is a budget, not a
hard assertion that emergence must be texture.

Identifiability safeguards are:

- strong multiscale loss on the grounded output itself;
- a zero-initialized, smoothly bounded emergent head;
- exact zero composite influence at `alpha=0`;
- separate head output, gradient, optimizer-update, contribution, and
  redundancy telemetry;
- independent grounded/emergent/composite artifacts and emergence sweeps.

The residual-fit term allows target-compatible detail to pass through the
emergent path, but its bounded magnitude and the independent grounded loss
prevent it from becoming an unconstrained reconstruction shortcut.

## Grounding objective

For antialiased area pooling `P_s`:

~~~text
L_ground = w0 |g-y|_1 + w1 |P_2(g)-P_2(y)|_1
         + w2 |P_4(g)-P_4(y)|_1 + 0.2 L_ssim_like
~~~

The default weights favor coarse and medium organization. Composite content,
palette, spatial-gradient structure, gamut, state, and memory terms remain
visible individually. Natural and crop targets do not receive a periodic seam
penalty; periodic texture targets do.

The age schedule uses smoothstep after the configured emergence start:

~~~text
r = clamp((age_phase - start) / ramp, 0, 1)
h = r^2 (3 - 2r)
ground(age)    = ground_strength * (1 - (1-ground_floor) h)
emergence(age) = emergence_strength * h
~~~

Grounding never decays to zero unless a user explicitly selects a synthesis
regime with a low floor.

## Compact world, native observations

Global and local observations share one normalized coordinate frame. The
balanced world remains 64x64/32x32. A whole 192px view supplies body-plan
supervision; later deterministic crops supply high-resolution observations
from a persistent Lanczos pyramid. A crop `[x, y, size]` queries the same
recurrent state and same renderer at the same global coordinates with a bounded
LOD signal.

The cross-resolution term compares a 2x render after proper area pooling with
the lower render. It includes direct, low-frequency, and edge agreement, but is
weighted lightly enough to permit legitimate subpixel detail.

The current aspect policy center-crops the largest source square and records the
transform. This avoids geometric stretch at low implementation cost. A
full-canvas/padding coordinate policy is deferred.

## Reference pathway and shortcut boundary

The pooled interface reference supports global reasoning. A separate learned
bias-free pointwise projection injects global or crop-local RGB into micro and
macro state updates. The drive is `tanh` bounded and multiplied by reference
fidelity; fidelity zero gives exact zero drive.

There is intentionally no renderer reference argument and no learned target
encoder in the flow path. Reference information must survive recurrent
development before becoming visible.

## Hierarchical recurrent interface

8x8 is the balanced reconstruction grid. For grids larger than 8x8, tokens are
pooled into a capped global interaction grid and combined with the local token
stream before writeback. This retains a compact global bottleneck and avoids
unrestricted quadratic attention at 16x16.

## Morphic growth and assimilation

Allocated `morph_layers` and active depth are distinct. Active depth is world
state, not constructor configuration. Reserved blocks are zero-contract
residuals and the stack applies one smooth projection after the active stack,
which makes activation function-preserving.

Supported checkpoint surgery is deliberately narrow: append a contiguous
suffix of physical MorphicBlocks with unchanged tensor shapes and semantics.
The loader inventories every tensor before assignment. It rejects silent
removal, rename, resize, malformed block names, noncontiguous indices, missing
old tensors, incomplete/nonfinite moments, and optimizer history for newborn
parameters.

Model, optimizer, world, and manifest carry one checkpoint transaction ID.
Manifest publication is last. One previous complete generation is retained via
hard links where supported (copy fallback) and restored automatically when the
current transaction cannot validate. Graft events store old/new anatomy, copied/new
parameters, preserved/new moments, birth generations, pre-graft state/loss
statistics, output fingerprint, and immediate preservation error.

Adaptive activation currently requires a configured interval, a complete loss
plateau window, low improvement, healthy finite/stability state, acceptable
clip scale, and reserve capacity. Deep seam half-life and generation-wise
activation/gradient curves remain an analysis extension; raw events are kept so
those measures can be added without losing history.

## Genuine optional conditional flow matching

Flow is isolated from the main phenotype objective. A fixed RGB-to-Oklab
transform defines endpoint `x1`; no learned target encoder exists. With
deterministic Gaussian `epsilon` and `t` sampled in the open unit interval:

~~~text
x_t = (1-t) epsilon + t x1
u_t = x1 - epsilon
L_CFM = E ||v_theta(x_t, t, c(S_mu,S_M,m))-u_t||^2
~~~

This is conditional rectified-flow velocity regression, not an NCA finite
difference heuristic. The compact head sees sampled recurrent fields, memory,
time, age, fidelity, emergence schedule, and LOD. It cannot accept target,
reference, genome, or endpoint-renderer outputs.

The normal composite is still the canonical result. Flow-only is experimental;
hybrid endpoint+flow is the recommended first test. Midpoint ODE sampling is
frozen-state, detached, deterministic, finite-checked, and analysis-only. v9
keeps flow global at 64px to preserve phone cost and avoid inconsistent crop
noise.

## Operational emergence diagnostics

v9 measures but does not automatically maximize residual energy, entropy,
novelty, or high-frequency power. Those can all be noise.

The analysis surface separates:

- grounding: multiscale content, palette, structure, SSIM-like and per-target
  statistics;
- emergence: bounded residual magnitude, band energy, total variation,
  contribution, persistence proxies, and emergence-strength frontier;
- coherence: image/state movement, saturation, seam, gamut, memory, and
  cross-resolution consistency;
- separability: fixed-seed/age pairwise output, low-band, edge, micro, macro,
  memory, and residual distances;
- autonomous behavior: frozen mature rollout, recurrence distances, drift,
  fingerprints, and conservative approximate-cycle candidates;
- perturbation recovery: deterministic micro/macro/memory noise plus localized
  micro and macro damage against an untouched control;
- render attribution: one frozen state with renderer paths changed;
- dynamical necessity: cloned trajectories with interface, scale, NCA, and
  physical mechanisms disabled.

The software reports observations as operational evidence. It does not assert
strong emergence, homeostasis, strange attractors, or causal macro control
without supporting diagnostics.

## S25 tradeoffs and remaining work

Heavy analysis runs only at final/checkpoint requests. Native pyramid levels
live on NAND rather than RAM. The always-allocated flow head makes the v9
checkpoint anatomy stable, but inactive flow has no gradients or weight decay.

Known design compromises:

- center-square supervision omits non-square border content;
- full native-detail flow and sparse local latent memory are deferred;
- target separability is explicit analysis rather than an always-on family
  sweep because repeated development of multiple targets is expensive;
- model statistics include weight/moment/participation summaries and normal
  windows include head gradient/update telemetry, but full per-layer activation
  hooks and expensive singular spectra are not always-on;
- grafting supports safe append-only MorphicBlocks, not magical width/channel
  resizing;
- decomposition labels are carried in JSON and sibling filenames rather than
  rasterized text, avoiding a font dependency on Termux.

These boundaries preserve a testable v9 core without making flow matching or
expensive analysis hold Reconstruction++ hostage.
