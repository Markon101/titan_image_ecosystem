# Fixed diffusion: noise contraction, patch recovery, and detail tradeoff

Fixed `nu=.01` substantially improves contraction of the tested macro noise,
but does not improve the erased-patch endpoint. At 512 steps, it also reduces
undamaged rendered edge RMS by 76.96% and RGB variance by 85.17%. This supports a
noise-damping effect, not a general repair or image-quality improvement. Do not
extend training on the strength of these results alone.

## Completed experiment

The 128-step panel used noise seeds 42 and 137. A single 512-step extension used
seed 42 after the two short runs agreed closely. Both operator arms and every
damage case start from the same frozen RMSNorm checkpoint: world step 128196,
saved age 4, target 88, micro 80x80 and macro 40x40, 32 channels. The checkpoint
was previously trained with differentiable normalization and the saturation
penalty; this is not an isolated normalization training comparison.

Each arm contains an undamaged control, macro-only uniform noise in [-.03,.03),
and a central-third macro patch erased in every channel. Patch damage is seed
independent, so repeating it checks reproducibility, not additional statistical
support. All evolution is CPU/f32 with a fixed saved genome, no runtime reference
tensors, fidelity zero, and zero optimizer updates. Damage is applied directly
to the young saved state without a guided burn-in. These conditions differ from
the earlier mature-state recovery report.

Protocol and invocation: [FROZEN_DIFFUSION_RECOVERY.md](FROZEN_DIFFUSION_RECOVERY.md).

## Recovery measurements

The state-distance ratio is the sum of micro, macro, and memory RMS distances
from the arm's own undamaged control, divided by its initial value. Lower is
closer to that control; 1 means the initial damage distance. It does not measure
target reconstruction or useful learned repair.

| Horizon | Case | No diffusion: final ratio | Fixed diffusion: final ratio |
|---|---|---:|---:|
| 128 | Noise, seed 42 | 1.194342 | 0.433083 |
| 128 | Noise, seed 137 | 1.168215 | 0.433913 |
| 128 | Patch, identical across seeds | 1.024665 | 0.954601 |
| 512 | Noise, seed 42 | 1.125844 | 0.068936 |
| 512 | Patch | 0.372461 | 0.407950 |

Noise's peak ratio for seed 42 decreases from 1.919608 to 1.286476 with diffusion.
The undiffused noise ratio falls to 0.662054 at step 320 before rising again to
1.125844 by step 512. A short contracting interval would miss this later increase;
it does not establish asymptotic stability or instability.

The predeclared sustained-contraction gate requires the ratio to stay below .5
at every step in the last quarter of the run. At 128 steps no case passes. For
the 512-step extension, evaluated over offsets 384 through 512 inclusive:

| Case | No diffusion: maximum tail ratio | Fixed diffusion: maximum tail ratio |
|---|---:|---:|
| Noise | 1.126769 | **0.111760** |
| Patch | 0.545125 | 0.591191 |

Only fixed-diffusion noise passes. Both patches finish below half their initial
distance but do not satisfy the entire final-quarter criterion. Fixed diffusion
leaves 9.53% more patch state distance and 27.87% more paired patch render L1 at
the 512-step endpoint. Those are measurements on this one saved condition, not
a general causal explanation of repair.

## What changes in the undamaged output

These comparisons use the undamaged trajectories, so they measure the effect of
diffusion itself, separately from its response to damage. All percentages are
fixed diffusion relative to no diffusion at the same offset.

| Measurement | 128 steps | 512 steps |
|---|---:|---:|
| Rendered edge RMS | -55.80% | -76.96% |
| Rendered RGB variance | +6.79% | -85.17% |
| Rendered mean | +5.90% | +12.84% |
| Micro low-band energy | -1.89% | -2.95% |
| Micro mid-band energy | -62.71% | -71.29% |
| Micro high-band energy | -90.98% | -93.69% |
| Raw control-to-control render L1 | 0.013282 | 0.019744 |

Low-band energy retention does not imply retained morphology. The stronger late
reduction in rendered variance is another reason to assess image detail along
with perturbation contraction. These statistics describe unmastered 96px renders;
they do not establish perceptual quality. `view_image` failed with the Termux
filesystem-sandbox error, so no visual inspection claim is made.

As a separate analytic check, periodic five-point diffusion alone on the 40x40
macro grid, with nu=.01 for 128 macro updates, retains 96.90% of a one-cycle
sinusoid's amplitude, 100% of a spatially constant component, and only 0.00232%
of a checkerboard component. This establishes the operator's strong scale
selectivity. It is not a prediction of the complete nonlinear trained map, nor
evidence that this particular patch residual is dominated by low frequencies.

## Verification and evidence

All three runs and an independent audit completed. Seven protected checkpoint/
config/metadata files and 45 source files retained their hashes. Both release
binaries still match the September 12 handoff; the executable code commit is
`7582c450a05ca30ad432ee22a9b7be37925f7fab`. No Rust source or training configuration
changed during this experiment. The binary manifest's dirty flag reflects the
pre-existing untracked local helpers and bytecode, with source hashes retained.

Across the three runs, 4,608 dynamics calls were checked for zero reference
drive, and zero optimizer updates occurred. Artifact hashes, row coverage,
world-step/age pairing, finite values, and normalized initial distances passed.
Undamaged diagnostics and patch trajectories/images match across the two short
seeds. The 128- and 512-step runs share six exactly matching numeric records and
18 byte-identical renders at offsets 0, 64, and 128. Twenty undamaged micro-field
diagnostic records also match the archived stage-two baseline/fixed-diffusion
trajectories exactly through offset 64.

Runtime was 159.7s and 169.8s for the two short seeds, and 590.5s for the long
extension: 15.3 minutes total, including startup. These are phone measurements,
not controlled hardware benchmarks.

Local numeric evidence (large artifacts remain outside Git):

- `analysis/frozen_diffusion_recovery_20260913/completed.json`
- `analysis/frozen_diffusion_recovery_20260913_long/completed.json`
- `analysis/frozen_diffusion_recovery_20260913/final_independent_audit.json`
- `analysis/frozen_diffusion_recovery_20260913/audit_recovery.py`
- `analysis/frozen_diffusion_recovery_20260913/stage2_comparison.json`
- `analysis/frozen_diffusion_recovery_20260913/horizon_prefix_comparison.json`
- `analysis/frozen_diffusion_recovery_20260913/diffusion_only_control.json`

[512-step contact sheet](../analysis/frozen_diffusion_recovery_20260913_long/seed_42/final_contact_sheet.png):
no diffusion on top, fixed diffusion below; columns are undamaged, noise, patch.

## Next direction

The next useful diagnostic is to split the residual by micro/macro/memory and,
for spatial fields, separate channel means from low/mid/high spatial bands. That
would test whether surviving damage is a coarse spatial deficit, a memory
response, or another mechanism before choosing a new learned intervention.

Before a learned-diffusion training fork, add a matched mature-state check and
an explicit detail/reconstruction retention gate. Keep autonomous dynamics
reference-free and guided held-out reconstruction separate. A learned coefficient
is still a hypothesis; these runs do not show that it repairs patches, retains
useful detail, or improves generalization. Do not combine other architecture or
objective changes with that prospective comparison.
