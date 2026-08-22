# TITAN Image Ecosystem v8

TITAN Image v8 is the recurrent-stability release of the phone-first,
multiscale image organism for the Galaxy S25 Ultra in Termux. It combines local
neural cellular automata, a looped global interface, bounded GRU/MorphicStack
memory, optional mathematical field operators, and an implicit high-resolution
renderer.

v8 is intentionally checkpoint-incompatible with v7 and writes only v8
artifacts. Existing files under /sdcard/Download/titan_image_v7 are never read
or overwritten. The default output root is /sdcard/Download/titan_image_v8.

The v8 boundary follows an artifact-backed v7 saturation postmortem: v7 fields
could reach 100% hard-clamp occupancy while recurrent-memory RMS exceeded 300,
and a coordinate/genome decoder could conceal the dead core. v8 bounds every
recurrent feedback path, restores the intended zero-output initialization,
adds differentiable stability barriers and a safe-stop watchdog, and reports
core-versus-decoder gradients. It remains an endpoint recurrent-development
model, not diffusion or flow matching.

## Start here

Build:

~~~sh
cargo build --release --locked
~~~

Start a balanced hybrid reconstruction/generation organism:

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v8 \
  --profile s25-balanced \
  --style alien-fluid \
  --mode family \
  --conditioning hybrid \
  --steps 1600 \
  --threads 8 \
  --run-tag morphic-rin-v8-01
~~~

Continue by repeating the exact training and architecture settings without
--fresh.

For a reconstruction-heavy experiment:

~~~sh
./target/release/titan_image \
  --fresh \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v8 \
  --profile s25-balanced \
  --style pure-nca \
  --mode family \
  --conditioning reconstruct \
  --reference-fidelity-max 0.95 \
  --steps 1600 \
  --threads 8 \
  --run-tag reconstruct-v8-01
~~~

For the matched autonomous control, use --conditioning generate. That removes
all reference pixels from recurrent-interface input while retaining the
content-keyed genome and ordinary image objective.

## Architecture

The balanced profile has:

- a 24-channel 64x64 micro field;
- a 24-channel 32x32 macro field updated every fourth development step;
- a 4x4 interface grid: only 16 global tokens;
- interface, GRU, and token width 128;
- three passes through one shared attention/feed-forward block;
- four physical morphic memory blocks, three active;
- a four-block 128-wide implicit renderer;
- 682,851 learned parameters.

Each world step pools the fields and RGB reference pyramid to the token grid,
adds genome/fidelity/age/memory conditioning, repeatedly applies shared
attention and feed-forward computation, updates GRU and morphic memory, and
uses zero-initialized spatial write heads to return information to both fields.
Local NCA and optional physical operators then evolve the dense fields.

This separates spatial size, learned capacity, and compute depth:

- --micro-size / --macro-size: spatial state;
- --interface-grid: global token count;
- --interface-width: recurrent representation width;
- --interface-loops: repeated computation with shared weights;
- --morph-layers: physical memory-block capacity;
- --morph-depth: active learned blocks.

Changing these controls selects a distinct checkpoint architecture. v8 does
not yet resize a saved checkpoint across them.

## Recurrent stability

The stable defaults are deliberately conservative:

- `dt=0.12`, `nca_gain=0.25`, and `state_leak=0.10`;
- `memory_limit=3.0` and `morph_residual_gain=0.02`;
- independent full memory reset at episode boundaries;
- a quartic smooth state projection instead of a hard clamp;
- bounded `tanh` interface writeback;
- state and memory soft-barrier weights of `0.02` and `0.002`;
- global gradient clipping at 1.0 with 48-update warmup;
- one full-core window for each decoder-only window in balanced/fast profiles.

`--core-update-every 2` means the renderer receives two updates for every one
core update: one full-core window and one decoder-only window. Use `1` when
both core and renderer should update every window.

The stability watchdog stops after eight consecutive windows above 25%
near-bound occupancy, saves a normal resumable checkpoint, marks
`stability_stopped=true`, and skips gallery rollout. Set
`--stability-patience 0` only for deliberate instability ablations.

Training logs and CSV rows report development steps/second, not optimizer
windows/second. The final metadata contains both average milliseconds per
development step and average development steps/second.

## Adjustable reference conditioning

--conditioning selects:

- generate: reference fidelity is always zero;
- hybrid: fidelity is deterministically sampled between the configured minimum
  and maximum, with null-reference windows from --reference-dropout;
- reconstruct: fidelity is fixed at --reference-fidelity-max.

The scalar is supplied to the network in addition to scaling the reference
pyramid. Hybrid mode is the intended foundation for a later image-to-image
sampling control. In this commit, fidelity is a training-window control;
render-only gallery generation remains reference-free.

## Hybrid Muon

AdamW remains the default. --optimizer hybrid-muon applies Muon only to 2D
matrices in recurrent-interface attention, feed-forward, and morphic blocks.
NCA, GRU, reference projections, write heads, renderer, biases, and
normalization parameters remain on AdamW.

~~~sh
--optimizer hybrid-muon --muon-momentum 0.95 --muon-ns-steps 5
~~~

Query, key, and value remain separate matrices. Optimizer kind and state are
checkpointed; changing optimizer requires a new tag and --fresh. Muon remains
experimental until matched wall-clock-controlled ablations earn it.

## ARM, OpenCL, and NPU boundary

The release build retains native AArch64 and FP16 instruction support. v8 also
precomputes cell-clock banks rather than allocating a host vector and Tensor at
every NCA step. Model and optimizer tensors remain FP32.

No OpenCL or QNN code is linked. The S25 exposes Qualcomm vendor libraries, but
the current Termux OpenCL loader enumerates zero platforms and direct vendor
loading is blocked by Android linker namespaces. QNN/HTP is a fixed-graph
inference deployment route, not a Candle autograd backend.

See RESEARCH_V7.md for the architecture and accelerator record and
RESEARCH_V8.md for the stability postmortem.

## Profiles

| Profile | Fields | Channels | Interface | Loops / morph | Core cadence | Train / output |
|---|---:|---:|---:|---:|---:|---:|
| s25-fast | 48 / 24 | 16 | 4x4x96 | 2 / L2 of 3 | 2 | 128 / 512 |
| s25-balanced | 64 / 32 | 24 | 4x4x128 | 3 / L3 of 4 | 2 | 192 / 768 |
| s25-quality | 80 / 40 | 32 | 5x5x160 | 4 / L4 of 6 | 1 | 256 / 1024 |

## Artifacts

v8 writes separately named model, optimizer, world, checkpoint-manifest,
metrics, metadata, raw/mastered image, state-atlas, snapshot, and gallery
artifacts with the titan_image_*_v8 prefix. World checkpoints now include
recurrent-interface memory. Model, world, optimizer kind/moments,
configuration signature, corpus fingerprint, and world step must agree before
continuation.

## Verification

~~~sh
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo build --release --locked
~~~

The end-to-end test covers fresh training, reference-pyramid input, recurrent
development, checkpoint output, and exact model/world/optimizer continuation.
