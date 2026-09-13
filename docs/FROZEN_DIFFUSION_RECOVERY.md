# Frozen diffusion recovery pilot

Completed results: [September 13 noise, patch, and detail comparison](FROZEN_DIFFUSION_RECOVERY_RESULTS_2026-09-13.md).

This testing-only mode compares no diffusion and fixed `nu=.01` using the same
CPU/f32 frozen sidecar as stage two. It adds no trainable parameters and never
saves model, world, or optimizer checkpoints. Both arms start from the exact saved
world, saved genome, and saved clocks, with no burn-in and no runtime references.

Each arm has three trajectories: undamaged, macro noise, and an erased macro
patch. Noise is ChaCha8 uniform `[-.03,.03)`, added to the macro field without a
new clamp. The patch erases the central one-third of each spatial dimension in
all macro channels. Only noise depends on the probe seed. Repeated patch cases
and undamaged controls check reproducibility; they are not independent samples.

```sh
PYTHONDONTWRITEBYTECODE=1 python -O scripts/frozen_diffusion_recovery.py \
  --config analysis/rmsnorm_followup_2026-09-09_matched/fork/config.json \
  --root analysis/frozen_diffusion_recovery_20260912 \
  --steps 128 --seeds 42 137
```

The binary's direct interface is `titan_develop --recovery true --config FILE
--output NEW --steps 128 --seed 42`. Recovery accepts only these options and is
limited to 512 steps. Other developmental diagnostics run separately. Output
roots must be new. A successful runner writes `completed.json` only after checking
artifacts, source/binary hashes, checkpoint immutability, reference absence, exact
initial renders across arms, and undamaged/patch reproducibility across seeds.
All validation gates remain active under optimized Python.

`recovery.jsonl` samples approximately eight intervals plus both endpoints. Each
record contains the sum of micro/macro/memory RMS distances from the corresponding
undamaged trajectory, divided by the initial damage distance; raw render L1 to
that paired control; undamaged image variance and edge RMS; and undamaged micro
spectral energies. Distances and final-quarter maximum ratios are computed at
**every** development step. A final-quarter maximum below .5 indicates sustained
paired contraction in that interval. It does not establish useful learned repair.

Renders are unmastered 96px RGB using the checkpoint's age-dependent emergence
schedule. `final_contact_sheet.png` has no-diffusion on the top row and fixed
diffusion on the bottom; columns are undamaged, macro noise, macro patch. Image
edge energy and latent spectral energy quantify changes but are not morphology
or image-quality scores. Lower paired image error can result from flattening.

The initial pilot uses one saved RMSNorm checkpoint at step 128196, age 4, target
88. It tests damage applied to that young saved state, not the older guided-burn-in
states from the earlier 512-step recovery report. Two noise seeds do not measure
across-target or across-checkpoint generalization. Autonomous target reconstruction
is not measured, and no target tensors enter the dynamics. Guided held-out
reconstruction should be evaluated separately before any training extension.

## Residual decomposition (schema v2)

Each sampled recovery row additionally exports `residuals`, ordered as noise,
patch. Each case reports micro, macro, and memory RMS, squared L2 energy, and its
contribution to the initial-distance-normalized recovery ratio. Field shares use
the sum of RMS values, so they add back to the existing recovery metric despite
different field sizes. Energy totals across fields are not used as RMS shares.

Spatial fields also report the damaged-minus-control channel means and a
partition into DC, low, mid, and high squared L2 energies. The per-channel means
are removed before the existing f64 DFT; radial cutoffs remain .125/.25 cycles
per original grid cell. Zero-energy fractions are null, and memory has no spatial
bands. Both Rust and the runner check partition closure and reconstruction of
the original distance. These passive measurements run only at existing output
intervals and do not change the model's f32 dynamics or checkpoint format.
