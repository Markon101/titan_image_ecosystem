# TITAN Image Ecosystem v5

TITAN Image is a phone-first laboratory for trainable morphogenic image dynamics. It combines a multirate neural cellular automaton (NCA), independently switchable mathematical operators, a lightweight periodic implicit renderer, persistent optimization, and deterministic genome interpolation. Its intended target is the Snapdragon 8 Elite in the Galaxy S25 Ultra running natively in Termux.

v5 is intentionally incompatible with every earlier image checkpoint. The v4 files under `/sdcard/Download/titan_image_v4` remain untouched; v5 defaults to `/sdcard/Download/titan_image_v5` and writes only `v5` artifacts.

## Start here

Build once:

```sh
cargo build --release --locked
```

Then start a new balanced fractal organism with this copy-pasteable command:

```sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v5 \
  --profile s25-balanced \
  --style fractal-flame \
  --mode texture \
  --steps 1600 \
  --threads 8 \
  --run-tag fractal-v5-01
```

`--fresh` is for the first invocation of a tag. `--steps` means additional development steps, not a final absolute step.

Continue the exact organism by repeating its training/architecture options without `--fresh`:

```sh
./target/release/titan_image \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v5 \
  --profile s25-balanced \
  --style fractal-flame \
  --mode texture \
  --steps 1600 \
  --threads 8 \
  --run-tag fractal-v5-01
```

The loader verifies schema, model/world/optimizer generation, every evolution and loss setting, tensor shapes, world step, and a content hash of the sorted corpus. Renaming a source does not change its genome; changing its bytes does invalidate exact continuation.

## What changed from v4

- Runtime-selectable micro/macro sizes, channels, genome width, NCA width, renderer width/depth, and coordinate bands.
- Near and dilation-2 perception rings, inspired by TITAN Audio v9's local/far communication, rather than a single local ring.
- One endpoint render/loss/backward per BPTT window. v4 rendered every recurrent step.
- Truly tape-free decoder-only windows: NCA weights are detached during those forwards instead of building and discarding core graphs.
- Cached periodic interpolation plans, Fourier coordinates, Laplacian kernels, and resized source tensors.
- Content-keyed genomes, deterministic epoch permutations, recursive-corpus and cache controls, and corpus-identity checkpoint validation.
- Fractal and quasiperiodic fields are stable target attractions `gain * (target - state)`, not constant velocities that drive channels into the clamp.
- A cyclic three-field oscillator joins reaction-diffusion, complex phase, IFS, and quasiperiodic operators.
- Normalized color correlation, normalized multi-lag spatial autocorrelation, log-contrast, and log-gradient losses replace scale-sensitive statistics that rewarded blur.
- A bounded style-specific state-to-color path prevents the renderer from ignoring the organism and solving the objective with coordinate bands alone.
- Global gradient clipping, AdamW warmup, configurable optimizer constants, richer diagnostics, peak-RSS reporting, build provenance, and a last-published checkpoint manifest.
- Raw/mastered output, micro/macro state atlases, deterministic developmental variants, a gallery contact sheet, and render-only exploration.

The v4 long run is useful negative evidence. Its final raw image had strong fixed decoder-lattice structure, gamut excess near `0.059`, and a phase profile of roughly 68 seconds dynamics, 424 seconds render/loss, and 860 seconds optimization over 1,500 steps. That is why v5 changes training geometry before micro-optimizing the already-cheap dynamics.

## Architecture

The balanced profile evolves a 24-channel `64x64` micro field every step and a `32x32` macro field every fourth step. Each NCA reads fixed identity/Sobel/Laplacian features at radius 1 and dilation 2, the macro context, and an eight-dimensional content-keyed genome. Two pointwise Swish layers mix those features before a zero-initialized update head and deterministic asynchronous cell clock.

Five separately weighted contributions form the derivative:

1. learned near/far NCA residual;
2. Gray-Scott-like reaction-diffusion in channels 0–1;
3. complex Ginzburg-Landau-like phase dynamics in channels 2–3;
4. bounded IFS/quasiperiodic target attraction in channels 4–5;
5. damped cyclic chemistry in channels 6–8.

Euler is the phone default; explicit midpoint remains available with `--integrator midpoint`. State is projected to a configurable compact interval after every step.

The renderer periodically upsamples micro and macro fields, adds low-amplitude global octave coordinates and genome conditioning, then applies a small residual pointwise network. Its learned OKLab-like head is combined with a bounded, style-specific projection of the actual state. That direct path closes the coordinate-only shortcut while retaining a trainable residual. Linear RGB gamut excursion is penalized before clipping, and display gamma is explicit.

The authoritative equations, stability boundaries, and non-claims are in [math.md](math.md). Metric meanings and the experiment protocol are in [METRICS.md](METRICS.md).

## S25 Ultra profiles

| Profile | Fields | Channels | NCA / renderer | Train / output | Core cadence | Observed role |
|---|---:|---:|---:|---:|---:|---|
| `s25-fast` | 48 / 24 | 16 | 64 / 48x2 | 128 / 512 | 1 in 8 windows | rapid style and seed search |
| `s25-balanced` | 64 / 32 | 24 | 96 / 64x3 | 192 / 768 | 1 in 4 windows | recommended sustained run |
| `s25-quality` | 80 / 40 | 32 | 128 / 96x4 | 256 / 1024 | 1 in 2 windows | slower, memory-heavy refinement |

Profiles are bases. Explicit flags override them regardless of argument order, so `--profile s25-fast --output-resolution 1024` keeps fast training but performs a larger final render.

The actual S25 can expose fewer cores after thermal or Android cpuset changes. v5 clamps the requested count to `available_parallelism()` and records both requested and effective values. In a short local sweep at 128px, eight effective threads beat seven and four; use `--threads 8` for a foreground run and try 4–6 when responsiveness, background survival, or thermal stability matters more than cold-start speed.

Measured on this checkout before the final documentation commit:

| Run | Work | Total | Peak RSS |
|---|---:|---:|---:|
| v4 baseline | 16 steps, 128px, 6 threads | 4.53 s | not recorded |
| v5 fast | 16 steps, 128px, 6 threads | 1.88 s | about 244 MiB |
| v5 fast thread sweep | 32 steps, 128px, 8 effective threads | 3.31 s | about 244 MiB |
| v5 balanced smoke | 16 steps, 192px, 8 effective threads | 5.12 s | about 719 MiB |

These are short on-device observations, not universal constants or a controlled thermal benchmark. v5 does more spatial supervision and has more parameters, so the v4/v5 row is an end-to-end workflow comparison rather than an isolated kernel speedup. Use the metadata phase profile after a thermal soak for decisions about a long run.

## Style bases

| Style | Dominant state-to-color geometry | Useful character |
|---|---|---|
| `alien-fluid` | phase + IFS + cyclic fields | flowing interference and saturated organic bands |
| `fractal-flame` | IFS field with phase/cyclic color | soft recursive triangular organisms |
| `reaction-garden` | activator/inhibitor + cyclic fields | Turing spots, fronts, and cellular gardens |
| `quasicrystal` | quasiperiodic + phase fields | long-beat interference and crystalline color |
| `pure-nca` | learned hidden channels | matched control with explicit operators removed |

Every style only sets ordinary controls. Override any gain after `--style`, or run `--list-presets` to inspect the current bases.

Suggested first experiments:

```sh
# Fast alien search
./target/release/titan_image --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v5 \
  --profile s25-fast --style alien-fluid --mode texture \
  --steps 800 --threads 8 --gallery 9 --run-tag alien-search-01

# Reaction-diffusion emphasis
./target/release/titan_image --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v5 \
  --profile s25-balanced --style reaction-garden --mode texture \
  --steps 1600 --threads 8 --run-tag reaction-v5-01

# Matched pure-learned control
./target/release/titan_image --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v5 \
  --profile s25-fast --style pure-nca --mode texture \
  --steps 800 --threads 8 --run-tag pure-control-01
```

Use distinct run tags for every seed, style, ablation, or architecture. This preserves the parent artifacts and makes comparisons unambiguous.

## Modes and corpus policy

- `single` requires exactly one source and uses spatial pixel L1 plus palette/gradient/gamut/seam terms.
- `family` cycles every source once per deterministic epoch, conditions on content-keyed genomes, and uses the same spatial objective.
- `texture` uses normalized color covariance and normalized autocorrelation at lags 1, 2, 4, and 8, plus palette/gradient/gamut/seam terms. It is the recommended mode for a heterogeneous collection.

Only PNG, JPEG, and WebP are admitted. File contents are decoded during preflight, generated names beginning with `titan_image_` are excluded, source tensors are retained up to `--image-cache`, and `--recursive-corpus` enables subdirectory traversal. The program never scans all of `/sdcard/Download` implicitly; `--corpus-dir` is mandatory.

## Render and explore without training

Use the exact architecture/training settings and corpus of the checkpoint, add `--render-only`, and freely change output-only controls:

```sh
./target/release/titan_image \
  --render-only \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v5 \
  --profile s25-balanced \
  --style fractal-flame \
  --mode texture \
  --threads 8 \
  --output-resolution 1536 \
  --gallery 9 \
  --gallery-steps 64 \
  --gallery-stride 16 \
  --run-tag fractal-v5-01
```

Gallery genomes interpolate two corpus genomes and add a small bounded mutation. Every variant starts from a deterministic fresh world and develops for `gallery_steps + variant_index * gallery_stride`, so the contact sheet samples both style space and developmental age. Render-only writes separate render metadata and does not modify model, optimizer, or world checkpoints.

## CLI surface

`./target/release/titan_image --help` is exhaustive. The major control families are:

- lifecycle: `--fresh`, `--render-only`, `--run-tag`, paths;
- compute: `--profile`, `--threads`, resolutions, BPTT/core/macro/cadence, cache;
- architecture: field sizes, channels, genome/NCA/renderer widths, renderer blocks, coordinate bands/gain;
- evolution: integrator, timestep, state bound, cell clock, NCA gain;
- physics: top-level gains and the Gray-Scott/complex-phase coefficients;
- optimization: learning rate, weight decay, Adam coefficients/epsilon, warmup, gradient clip;
- objective: content, palette, structure, seam, and gamut weights;
- appearance: style, state skip, chroma, gamma, mastering strength;
- exploration: gallery count, development steps/stride/seed and state-atlas output;
- ablations: `--no-reaction-diffusion`, `--no-complex-phase`, `--no-fractal`, `--no-quasiperiodic`, `--no-cyclic`, and `--no-mastering`.

## Artifacts

With `--run-tag fractal-v5-01`, the output directory contains:

- `titan_image_model_v5_fractal-v5-01.safetensors`
- `titan_image_optimizer_v5_fractal-v5-01.safetensors`
- `titan_image_world_v5_fractal-v5-01.safetensors`
- `titan_image_checkpoint_v5_fractal-v5-01.json`
- `titan_image_metrics_v5_fractal-v5-01.csv`
- `titan_image_run_metadata_v5_fractal-v5-01.json`
- `titan_image_render_metadata_v5_fractal-v5-01.json` after render-only use
- `titan_image_raw_v5_fractal-v5-01.png`
- `titan_image_mastered_v5_fractal-v5-01.png`
- `titan_image_micro_state_v5_fractal-v5-01.png`
- `titan_image_macro_state_v5_fractal-v5-01.png`
- numbered raw/mastered variant PNGs and `titan_image_gallery_v5_fractal-v5-01.png`
- periodic `titan_image_snapshot_v5_fractal-v5-01_*.png` files when enabled

Raw images are architectural evidence. Mastered images apply toroidal local contrast, bloom, and a shoulder. State atlases normalize every channel independently and are diagnostic maps, not comparable color values or artworks.

## Research basis and boundaries

The coarse-state/implicit-decoder direction follows [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/abs/2506.22899). Genome conditioning and interpolation follow the multi-texture direction of [Signal Responsive NCA](https://arxiv.org/abs/2407.05991). Noise and discretization are treated as experiment variables in the spirit of [NoiseNCA](https://arxiv.org/abs/2404.06279). The explicit contractive component is informed by [Learnable Fractal Flames](https://arxiv.org/abs/2406.09328) and [Differentiable Iterated Function Systems](https://arxiv.org/abs/2203.01231).

TITAN Image has recurrent latent state, but it has no actions, observation encoder, transition dataset, or planning evaluation. It is an autonomous generative dynamical system, not an action-conditioned world model. A finite output containing IFS structure is not proof that the whole image is a mathematical fractal. The project records these non-claims deliberately.

## Validation

```sh
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo build --release --locked
```

The end-to-end test covers fresh training, backpropagation, atomic PNG and checkpoint output, exact optimizer/world continuation, corpus validation, metrics, and metadata.
