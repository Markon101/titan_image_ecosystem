# Titan Image OpenCL notes

Status: Phase 2A forward MVP. CPU/Candle remains default and numerical oracle. Checkpoint tensor names, schema v9, signatures, training dynamics, losses, and serialization unchanged.

## Implemented

- Optional `opencl` Cargo feature using `opencl3` 0.12.3 with runtime dynamic loading.
- `--compute-backend cpu|opencl|auto`; default `cpu`.
- Frozen render-only/analysis/probe renderer MLP on OpenCL FP32. Input features upload once; input layer, all residual blocks, and both output heads remain device-resident; six output channels download once. Weights upload once and remain resident.
- Frozen FP32 micro and macro NCA forward path on OpenCL: toroidal two-ring perception, input/residual/output pointwise transforms, swish/tanh, deterministic clock mask, and NCA gain. Weights, clock masks, and resolution-sized work buffers remain resident.
- CPU retains interface attention/GRU/MorphicStack, reference drives, integration/state projection, physical operators, renderer preprocessing/postprocessing, metrics, losses, PNG, scheduling, checkpoints, and all backward operations.
- `opencl` training request fails clearly. `auto` training remains CPU. Missing OpenCL under `auto` falls back to CPU. No stale GPU weights during training.
- Device report: name/vendor/version/OpenCL C, compute units, workgroup size, global/max-allocation/local memory, FP16 extension.

CLBlast: no installed `libCLBlast`; available Rust binding is old. Custom Titan-specific renderer and NCA kernels were smaller and produced a working baseline. Revisit CLBlast or tiled GEMM only after profiling.

## Adreno 830 runtime

Termux ICD requires `OCL_ICD_ASSUME_ICD_EXTENSION=1`: Qualcomm driver exposes `clIcdGetPlatformIDsKHR` but not global `clGetPlatformInfo`, so stock `ocl-icd` otherwise skips it.

Detected: `QUALCOMM Adreno(TM) 830`; OpenCL 3.0; Qualcomm build `0800.64.7`; 12 compute units; max workgroup 1024; 5,556 MiB global; 1,024 MiB max allocation; 32 KiB local; native `cl_khr_fp16` (unused; FP32 only).

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

## Benchmarks

Uncontrolled background conditions: user was using other apps. Directional only; rerun foreground and thermally stabilized.

| Workload | CPU | OpenCL | speedup |
|---|---:|---:|---:|
| saved-checkpoint renderer 192px | 673.798 ms | 282.928 ms | 2.382x |
| saved-checkpoint renderer 384px | 2573.414 ms | 1105.272 ms | 2.328x |
| full render-only process 768px | 15.745 s | 12.772 s | 1.233x |
| saved c303 Pure-NCA dynamics, 32 steps | 4989.155 ms | 3028.459 ms | 1.647x |

The 768px process includes corpus startup, checkpoint loading, CPU feature/postprocessing, PNG encoding, and OpenCL program compilation. Renderer-only test initializes OpenCL before timing.

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
```

## Limits / next work

Phase 2 is partial; Phase 3 is not attempted. NCA state and context are uploaded and the delta downloaded once per active scale per step because interface attention/GRU/MorphicStack and integration remain CPU. The expensive NCA work buffers are cached, but the organism is not yet continuously GPU-resident. Training remains CPU and explicit OpenCL training is rejected. Renderer feature assembly still creates a large CPU tensor and renderer buffers are reallocated per render. Runtime errors after successful initialization propagate; only initialization unavailability auto-falls back.

Highest-value next tasks:

1. Complete Phase 2 residency: port reference drives, interface attention/GRU/MorphicStack, macro upsampling, Euler/midpoint integration, and smooth state projection; download only requested metrics/renders.
2. Share one OpenCL context/queue across renderer and dynamics; tile dense kernels or integrate CLBlast; add foreground thermally stabilized 32/64/128-step benchmarks.
3. Keep Phase 3 gated until full frozen forward parity is sound; then implement one deterministic training window, gradients, clipping, and parameter-delta parity.
