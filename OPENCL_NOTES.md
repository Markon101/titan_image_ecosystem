# Titan Image OpenCL notes

Status: Phase 1 MVP. CPU/Candle remains default and numerical oracle. Checkpoint tensor names, schema v9, signatures, dynamics, losses, and serialization unchanged.

## Implemented

- Optional `opencl` Cargo feature using `opencl3` 0.12.3 with runtime dynamic loading.
- `--compute-backend cpu|opencl|auto`; default `cpu`.
- Frozen render-only/analysis/probe renderer MLP on OpenCL FP32. Input features upload once; input layer, all residual blocks, and both output heads remain device-resident; six output channels download once. Weights upload once and remain resident.
- CPU retains field upsampling, coordinate/genome feature construction, emergent filtering, state skip, OKLab conversion, metrics, losses, PNG, scheduling, checkpoints, and all dynamics.
- `opencl` training request fails clearly. `auto` training remains CPU. Missing OpenCL under `auto` falls back to CPU. No stale GPU weights during training.
- Device report: name/vendor/version/OpenCL C, compute units, workgroup size, global/max-allocation/local memory, FP16 extension.

CLBlast: no installed `libCLBlast`; available Rust binding is old. Custom renderer-specific dense kernels were smaller and produced a working baseline. Revisit CLBlast or tiled GEMM only after profiling.

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

## Benchmarks

Uncontrolled background conditions: user was using other apps. Directional only; rerun foreground and thermally stabilized.

| Workload | CPU | OpenCL | speedup |
|---|---:|---:|---:|
| saved-checkpoint renderer 192px | 673.798 ms | 282.928 ms | 2.382x |
| saved-checkpoint renderer 384px | 2573.414 ms | 1105.272 ms | 2.328x |
| full render-only process 768px | 15.745 s | 12.772 s | 1.233x |

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
```

## Limits / next work

Phase 2 and Phase 3 not attempted. Development dynamics and backward remain CPU. Feature assembly still creates a large CPU tensor; device buffers are reallocated per render. Attribution uses the accelerated learned heads but CPU postprocessing. Runtime errors after successful initialization currently propagate; only initialization unavailability auto-falls back.

Highest-value next tasks:

1. Cache resolution-sized device buffers; fuse periodic field sampling, coordinates, state skip, residual filtering, and color conversion. Upload micro/macro/genome; download final outputs only.
2. Profile/tile dense kernels or integrate maintained CLBlast directly; add warm/cold and thermally stabilized 192/384/768 benchmark runs.
3. Phase 2 frozen forward path: FP32 NCA perception plus pointwise transforms, then interface/morph operations; parity at short/long ages before any backward work.
