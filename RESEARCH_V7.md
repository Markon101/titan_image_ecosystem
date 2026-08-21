# TITAN Image v7 architecture research record

This file records why v7 uses a multiscale recurrent interface and which ideas
remain experimental.

## Implemented basis

TITAN Audio v9 supplies the project-specific precedent: multirate local fields,
near/far perception, recurrent memory, morphic capacity, spatial-token
decoding, bounded phone memory, and evidence-gated structural changes.

External architecture references:

- Recurrent Interface Networks: https://arxiv.org/abs/2212.11972
- Frequency-Time Diffusion with NCA: https://arxiv.org/abs/2401.06291
- Universal Transformers: https://arxiv.org/abs/1807.03819
- SinDDM: https://arxiv.org/abs/2211.16582
- Multi-Scale Latent Factorization: https://arxiv.org/abs/2501.13349

RIN is the closest global-computation match: route dense data through a small
latent interface, perform global attention there, and return information to the
data field. v7 combines this with local NCA rather than attending over every
image cell. SinDDM supports the low-data, scale-conditioned direction; it does
not imply v7 currently implements diffusion.

## Reconstruction and future transport

The reference pyramid and explicit continuous fidelity scalar support
deterministic reconstruction, partially conditioned variation, null-reference
development, and later flow/diffusion transport.

The next objective candidate is rectified flow:

x_t = (1-t) epsilon + t x_0

u_t = x_0 - epsilon

L_flow = E ||v_theta(x_t,t,c)-u_t||^2.

A conventional multistep flow should establish the baseline before MeanFlow,
consistency distillation, or one-step claims.

References:

- SiT: https://arxiv.org/abs/2401.08740
- MeanFlow: https://arxiv.org/abs/2505.13447
- SDEdit: https://arxiv.org/abs/2108.01073
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598

## Muon boundary

Muon reference: https://kellerjordan.github.io/posts/muon/

Recent vision and diffusion-transformer evidence:

- https://arxiv.org/abs/2605.24770
- https://arxiv.org/abs/2608.02502

The latter reports harmful coupling when distinct QKV/AdaLN subspaces are
orthogonalized together. v7 keeps query, key, and value separate and applies
Muon only to selected hidden matrices. The evidence is recent and does not
replace a TITAN-specific ablation.

## ARM observations

Measured v6 profiles were dominated by backward and endpoint render/loss, not
optimizer stepping. Priorities are eliminating repeated layout copies, fusing
periodic perception, caching clocks, testing selective FP16 only after parity
checks, and splitting kernel/copy/allocation timing. Cached clocks are
implemented in v7. Native flags do not turn FP32 tensors into FP16.

## OpenCL observation

On the development phone:

- product SM-S938U, SoC SM8750;
- Qualcomm vendor OpenCL libraries are installed;
- the Termux Khronos loader is installed;
- clinfo currently reports zero platforms;
- direct vendor preloading is blocked by the Android linker namespace.

References:

- https://github.com/KhronosGroup/OpenCL-ICD-Loader
- https://docs.qualcomm.com/doc/80-NB295-11/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf

No design should depend on OpenCL until a standalone probe enumerates the
device and validates resident vector, matrix, and periodic-convolution kernels.
Whole forward/backward islands must remain resident; per-operation CPU/GPU
bouncing is not viable.

## NPU observation

QNN HTP v79 runtime, stub, skeleton, and system libraries are installed.
QNN/QAIRT is a fixed-graph inference deployment path, not a Candle training
backend. Android NNAPI is deprecated on Android 16/API 36.

Potential uses are a fixed-loop sampler, reference encoder, renderer, or
distilled few-step model. Variable morphic capacity and dynamic loops require
fixed compiled profiles or CPU orchestration.
