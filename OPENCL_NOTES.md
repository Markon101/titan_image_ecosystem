# Titan Image OpenCL notes

Status: Phase 3A hybrid-training MVP. CPU/Candle remains default and numerical oracle. Checkpoint tensor names, schema v9, signatures, schedules, losses, clipping, AdamW state, and serialization remain unchanged.

## Implemented

- Optional `opencl` Cargo feature using `opencl3` 0.12.3 with runtime dynamic loading.
- `--compute-backend cpu|opencl|auto`; default `cpu`.
- Frozen render-only/analysis/probe renderer MLP on OpenCL FP32. Input features upload once; input layer, all residual blocks, and both output heads remain device-resident; six output channels download once. Weights upload once and remain resident.
- Frozen FP32 micro and macro NCA forward path on OpenCL: toroidal two-ring perception, input/residual/output pointwise transforms, swish/tanh, deterministic clock mask, and NCA gain. Weights, clock masks, and resolution-sized work buffers remain resident.
- CPU retains interface attention/GRU/MorphicStack, reference drives, integration/state projection, physical operators, renderer preprocessing/postprocessing, metrics, losses, PNG, scheduling, checkpoints, and full-core autograd windows.
- Stage 3A explicit `opencl` training: detached decoder windows use OpenCL NCA forward, renderer forward, and manual renderer backward. Tiled FP32 weight-gradient kernels feed the unchanged clipping/AdamW optimizer. Full-core BPTT windows remain CPU/Candle. Cached NCA and renderer weights refresh after every optimizer update.
- `auto` training remains CPU. Missing OpenCL under `auto` falls back to CPU.
- Device report: name/vendor/version/OpenCL C, compute units, workgroup size, global/max-allocation/local memory, FP16 extension.

CLBlast: no installed `libCLBlast`; available Rust binding is old. Custom Titan-specific renderer/NCA kernels and a 16x16 tiled renderer weight-gradient kernel produced the working baseline. Revisit CLBlast for full-core backward and larger GEMMs.

## Adreno 830 runtime

Termux ICD requires `OCL_ICD_ASSUME_ICD_EXTENSION=1`: Qualcomm driver exposes `clIcdGetPlatformIDsKHR` but not global `clGetPlatformInfo`, so stock `ocl-icd` otherwise skips it.

Detected: `QUALCOMM Adreno(TM) 830`; OpenCL 3.0; Qualcomm build `0800.64.7`; 12 compute units; max workgroup 1024; 5,556 MiB global; 1,024 MiB max allocation; 32 KiB local; native `cl_khr_fp16` (unused; FP32 only).

Verified Termux exposure for this phone:

```sh
pkg install ocl-icd clinfo opencl-headers
mkdir -p "$PREFIX/lib/titan-opencl" "$PREFIX/etc/OpenCL/vendors"
cp /vendor/lib64/libOpenCL_adreno.so "$PREFIX/lib/titan-opencl/"
printf "%s\n" "$PREFIX/lib/titan-opencl/libOpenCL_adreno.so" > \
  "$PREFIX/etc/OpenCL/vendors/adreno.icd"
export OCL_ICD_ASSUME_ICD_EXTENSION=1
clinfo | sed -n "1,80p"
```

The app-readable driver copy matters on Android linker namespaces; adding `/vendor/lib64` to `LD_LIBRARY_PATH` alone is not reliable. `OCL_ICD_ASSUME_ICD_EXTENSION=1` is documented by [OCL-ICD](https://github.com/OCL-dev/ocl-icd/blob/master/doc/libOpenCL.7.txt.in) for ICDs that do not globally expose the full extension entry points. Android namespace restrictions are documented by [AOSP](https://source.android.com/docs/core/architecture/vndk/linker-namespace).

## Correctness

CPU default reproduced the existing 768px `v9-grounded-emergent-c81-01` raw PNG byte-for-byte: SHA-256 `dabbccd070fbb2bf87a64ff7a5d75fb78f8c02b5d71fe62a1d4c06a5bd1d6837`.

Synthetic deterministic 24px renderer parity: max abs `0.00000033`, mean abs `<0.00000001`, RMS `0.00000001`.

Saved `long-run-readiness` checkpoint/world, FP32:

| Resolution | max abs | mean abs | RMS |
|---:|---:|---:|---:|
| 192 | 0.00019872 | 0.00000005 | 0.00000062 |
| 384 | 0.00003764 | 0.00000005 | 0.00000018 |

Saved `v9-grounded-emergent-c81-01` end-to-end 768px raw PNG: maximum quantized difference `1/255`; normalized mean absolute difference `3.54598e-8`; normalized RMSE `1.17923e-5`.

Phase 2A synthetic NCA delta: max abs `1e-8`. Synthetic 32-step dynamics: age 1/8/32 max abs `1e-8`/`1e-8`/`6e-8`. Saved `v9-pure-nca-fat-c303-01`, same checkpoint world, target, references, genome, and fidelity: age +1/+8/+32 state max abs `6e-8`/`2.4e-7`/`3.6e-7`; final 192px render max abs `6e-7`, mean abs `3e-8`, RMS `6e-8`.

Stage 3A renderer backward primitive: max gradient drift `1.2e-7`. Deterministic two-window tiny training test (one CPU full-core plus one OpenCL decoder window): loss/state byte-equivalent at float precision, max parameter drift `1e-8`, final PNG byte-identical after cache refresh. Saved c303 two-window continuation: loss equal to seven decimals, model max drift `3e-8`, optimizer max drift `1.0e-7`, world max drift `7.6e-7`; final 192px PNG max difference `1/255`, normalized mean `7.09e-8`.

## Benchmarks

Uncontrolled background conditions: user was using other apps. Directional only; rerun foreground and thermally stabilized.

| Workload | CPU | OpenCL | speedup |
|---|---:|---:|---:|
| saved-checkpoint renderer 192px | 673.798 ms | 282.928 ms | 2.382x |
| saved-checkpoint renderer 384px | 2573.414 ms | 1105.272 ms | 2.328x |
| full render-only process 768px | 15.745 s | 12.772 s | 1.233x |
| saved c303 Pure-NCA dynamics, 32 steps | 4989.155 ms | 3028.459 ms | 1.647x |
| saved c303 detached decoder window, 4 steps | 3438.550 ms | 2178.545 ms | 1.578x |

The Stage 3 decoder-window timing excludes startup and final rendering; its surrounding process timing was invalidated by severe background load. The 768px process includes corpus startup, checkpoint loading, CPU feature/postprocessing, PNG encoding, and OpenCL program compilation. Renderer-only test initializes OpenCL before timing.

## Build/run

```sh
cargo build --release --locked --features opencl
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/titan_image \
  --corpus-dir /sdcard/Download/titan_image_sources \
  --output-dir /sdcard/Download/titan_image_v9 \
  --run-tag v9-grounded-emergent-c81-01 \
  --profile s25-balanced --style alien-fluid \
  --research-preset grounded-emergent --seed 42 \
  --steps 100000 --threads 8 --episode-steps 64 --bptt 4 \
  --image-cache 81 --snapshot-every 12 --checkpoint-every 1000 --log-every 10 \
  --gallery 0 --no-emergence-gallery --no-state-atlas \
  --render-only --compute-backend opencl
```

Hybrid Stage 3A resume of the current Pure-NCA checkpoint:

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/titan_image \
  --corpus-dir /sdcard/Download/titan_image_sources_nca256/ \
  --output-dir /sdcard/Download/titan_image_v9_pure_nca \
  --run-tag v9-pure-nca-fat-c303-01 \
  --profile s25-balanced --style pure-nca --research-preset grounded-emergent \
  --seed 42 --steps 50000 --threads 6 \
  --micro-size 80 --macro-size 40 --channels 32 --ca-hidden 160 \
  --interface-grid 8 --interface-width 160 --interface-loops 4 \
  --morph-layers 8 --morph-depth 4 --morph-max-depth 8 \
  --render-hidden 160 --render-blocks 5 --train-resolution 192 \
  --episode-steps 64 --bptt 4 --image-cache 128 --terminal rich \
  --snapshot-every 64 --checkpoint-every 2000 --log-every 10 \
  --compute-backend opencl
```

Graceful fallback: use `--compute-backend auto`. CPU-only build remains `cargo build --release --locked`.

Focused tests:

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 TITAN_OPENCL_TEST=1 \
  cargo test --features opencl opencl_renderer_matches_cpu -- --nocapture

OCL_ICD_ASSUME_ICD_EXTENSION=1 \
TITAN_OPENCL_CHECKPOINT_MODEL=/path/to/titan_image_model_v9_TAG.safetensors \
TITAN_OPENCL_CHECKPOINT_WORLD=/path/to/titan_image_world_v9_TAG.safetensors \
  cargo test --release --features opencl \
  opencl_checkpoint_renderer_parity_and_benchmark -- --nocapture

OCL_ICD_ASSUME_ICD_EXTENSION=1 \
TITAN_OPENCL_DYNAMICS_METADATA=/path/to/titan_image_run_metadata_v9_TAG.json \
TITAN_OPENCL_DYNAMICS_MODEL=/path/to/titan_image_model_v9_TAG.safetensors \
TITAN_OPENCL_DYNAMICS_WORLD=/path/to/titan_image_world_v9_TAG.safetensors \
  cargo test --release --features opencl \
  opencl_checkpoint_pure_nca_rollout_and_render -- --nocapture

OCL_ICD_ASSUME_ICD_EXTENSION=1 TITAN_OPENCL_TEST=1 \
  cargo test --features opencl renderer_backward_matches_candle -- --nocapture

OCL_ICD_ASSUME_ICD_EXTENSION=1 TITAN_OPENCL_TEST=1 \
  cargo test --features opencl opencl_decoder_training_matches_cpu -- --nocapture
```

## Limits / next work

Phase 2 and Phase 3 remain partial. NCA state/context are uploaded and delta downloaded once per active scale because interface attention/GRU/MorphicStack and integration remain CPU. Full-core BPTT/backward remains CPU; only detached decoder windows use manual OpenCL backward. Renderer training activations remain GPU-resident for one window, but renderer features originate on CPU. Runtime errors after successful initialization propagate; only initialization unavailability auto-falls back.

Highest-value next tasks:

1. Complete Phase 2 residency: port reference drives, interface attention/GRU/MorphicStack, macro upsampling, integration, and state projection; download only metrics/renders.
2. Extend Stage 3 through one full-core BPTT window: NCA/interface backward plus state-gradient recurrence, retaining CPU clipping/AdamW oracle parity.
3. Share one OpenCL context/queue, evaluate CLBlast for full-core GEMMs, then run foreground thermally stabilized 32/64/128-step training benchmarks.
