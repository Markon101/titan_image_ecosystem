# Frozen developmental diagnostics (schema titan.development.v1)

`titan_develop` is a separate CPU analysis binary. The training CLI, model tensor
registry, checkpoint signatures, optimizer layout and existing output paths are
unchanged. It imports a complete current checkpoint using the strict read-only
loader; it cannot recover/republish an older generation, train, or save a checkpoint.
Use a quiescent checkpoint or an immutable copy. SHA-256 checks before import,
after import and after analysis detect concurrent changes; they are not a lock or
an atomic snapshot. A changed input invalidates completion.

```sh
cargo build --release --locked --bin titan_develop
./target/release/titan_develop \
  --config analysis/rmsnorm_followup_2026-09-09_matched/parent/config.json \
  --output analysis/development_baseline \
  --steps 16 --perturb-epsilon .02 --perturb-band high \
  --response-ages 0,8,16 --response-epsilons .01,.02 \
  --response-bands low,mid,high
```

`--config` accepts a full RunConfig or a metadata JSON containing `config`.
`--output` must not exist; its parent must exist. Corpus caches are redirected
into this new directory. The source config and effective CPU config are saved.
`--help` lists all controls. Probe ages are offsets from the saved developmental
age, not fresh-organism ages. The saved target index selects a fixed genome;
both target reference tensors are absent and fidelity is zero throughout.
No target reconstruction loss or image rendering is performed.

## State and definitions

The state vector concatenates micro `[1,C,Hm,Wm]`, macro `[1,C,Ha,Wa]`, and
interface memory `[1,M]`, in contiguous tensor order. Distances use all these
scalars without rescaling the grids or channels. Counters/anatomy are retained
in the world but excluded from numerical norms. Since legacy dynamics depend on
clocks and macro cadence, comparisons use identical clocks. Growth is conditional
on that forcing sequence; a nearly unchanged state is not an autonomous fixed point.

* Spatial state energy: `sum(x²)/(2HW)`, with channels summed at each site.
  Full-state energy is separately named `full_state_energy_per_scalar`,
  `sum(x²)/(2n)`, where `n=C(HmWm+HaWa)+M`.
* Update: unnormalized full-state Euclidean `U=||x(t)-x(t-1)||₂`;
  update energy `U²/2`, update RMS `U/sqrt(n)`. Initial updates are null.
* Population variance over all field scalars and separately over each channel's
  spatial sites. Hidden memory L2 is reported separately.
* Spatial gradient norm: centered first differences in both directions, unit grid
  spacing, periodic boundaries, summed across all channels. This operator has
  zero response at the Nyquist checkerboard, so use high-band energy alongside it.
  Parameter-gradient norm is null: this binary does not define a loss/backward pass.
* Spectrum: separable **f64 DFT**, original grid, periodic domain, no windowing or
  downsampling. Radius is `sqrt(fx²+fy²)`, with
  `fx=min(kx,W-kx)/W`, `fy=min(ky,H-ky)/H`, in cycles per grid cell.
  Low is `[0,low_cutoff]`, mid `(low_cutoff,mid_cutoff]`, high the remainder;
  defaults are `.125` and `.25`. These cutoffs are cell-relative on each grid,
  not common physical wavelengths across grids. DC is part of low energy.
  Parseval band energy is `sum_band |DFT(x)|²/(HW) = ||P_band x||²`.
* Finite-time growth: `(log(delta_T/delta_0))/T` without renormalization.
  Perturbations affect micro only; growth measures micro+macro+memory.
  Actual nonzero initial distance after f32 conversion is the denominator.
  At T=0 the rate is null. Exact coalescence is flagged, with null rate
  (mathematically negative infinity); it is never replaced with an epsilon floor.
* Recurrence: `Dij=||xi-xj||₂`, full retained states, not projections.
  `recurrence.f64le` is row-major little-endian f64; `recurrence.json` supplies
  dimensions and actual ages. Storage includes both endpoints. Sample capacity
  and a 512 MiB retained-state ceiling are enforced; increase stride for long runs.
* Nonlinear response: identical-state, identical-clock evaluations of G, G+ and G-.
  Seeded uniform noise is projected into a chosen radial band, per-channel DC
  removed, then globally normalized to micro L2=1. Direction is reused across ages
  and epsilon values. `Q=LP(G+ + G- - 2G)/(2 epsilon²)` uses f64 subtraction of f32
  model outputs. LP covers micro and macro separately, excludes nonspatial memory.
  Norms are reported separately and combined; the ratio denominator is the full
  baseline update including memory. A zero denominator gives null.
  Actual positive/negative input norms and symmetry residual expose f32 rounding.
  Compare multiple epsilons: a divergent tiny-epsilon response may be roundoff.
* Cancellation framework: `C=(||R||+||T||+||D||+||S||)/(||R+T+D+S||+1e-12)`.
  In this pass **R means the entire effective legacy update** `Glegacy(x)-x`,
  including existing physical operators, memory, integrator and limiter. It does
  not isolate internal learned NCA contributions or reveal cancellation inside G.
  T, D and S are initially zero. Zero total activity yields C=0.

## Artifacts and interpretation

`trajectory.jsonl` streams every step. `response.jsonl` contains only selected
age/band/epsilon probes. No matrices or tensor dumps are printed to the terminal.
`temporal.json` saves biased autocorrelation of full-state energy (up to 256 lags)
and an unwindowed, mean-centered temporal DFT of at most the last 1024 steps.
Constant series have null autocorrelation. Frequencies are cycles per step;
scalar periodicity can alias state motion and does not classify an attractor.
`state_stagnation_candidate` requires a configurable consecutive window of small
full-state update RMS values. It is deliberately a descriptive candidate only.

`manifest.json` records schema, binary SHA-256, build commit/dirty flag/Rust flags,
source/effective full config, seeds, exact checkpoint/model/world/optimizer hashes,
optimizer load report, saved training step/age, conditioning and perturbation policy.
`summary.json` is written only on success and hashes all diagnostic artifacts.
A manifest alone means incomplete; its `complete:false` is never overwritten.
The summary confirms protected training-file hashes are unchanged.

Runtime buckets separate base dynamics, diagnostic computation and additional
probe evaluations. The DFT costs O(C*HW*(H+W)), with O(C*HW) scratch; it favors
an auditable dependency-free first pass over FFT speed. Recurrence retains
O(K*n) f64 values and computes O(K²*n) distances. Its 512 MiB limit excludes
model/corpus/spectral scratch and is not a total-process memory limit.

This first pass does not add trainable checkpoint tensors, stress closure,
channel timescale groups, scale schedules, self-similar event extraction or
metastability classification. Those need separate ablations and validation.
No entropy/chaos objective, attractor correction or new state clamp is introduced.
