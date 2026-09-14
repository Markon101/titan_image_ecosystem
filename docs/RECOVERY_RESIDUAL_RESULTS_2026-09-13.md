# Recovery residual decomposition — September 13

The erased-patch residual is primarily a macro-field problem, but fixed diffusion's
worse *total* patch endpoint is accounted for by increased micro and memory error.
This narrows the next experiment to separating where diffusion is applied, before
introducing a learned coefficient or extending training.

## Implementation and verification

Code commit `3d7a1aa` adds passive sampled micro/macro/memory RMS contributions and,
for spatial fields, channel-mean/DC plus centered low/mid/high squared-L2 energy.
Field shares reconstruct the existing sum-of-RMS distance; spatial energy shares
partition each field separately. These are different normalizations. Zero-energy
fractions are null. Model dynamics, objectives, and checkpoint layout are unchanged.

Validation passed: 15 focused Rust tests, two optimized-Python validation tests,
strict Clippy, formatting, and the completed 512-step seed-42 replay. All 18 legacy
sample records, the legacy recovery summary, and all 55 PNGs match the previous
512-step run exactly. Maximum relative spatial energy closure error was
`9.08097e-13`. Protected checkpoint files and the analysis source snapshot retained
their hashes. There were zero optimizer updates and 3072 reference-free dynamics
calls across the six trajectories.

The first attempt stopped without a completion receipt after logging step 416.
Its partial output is excluded. An unrelated workspace edit to `src/run_lease.rs`
appeared separately. The completed retry used an exported `3d7a1aa` source snapshot
and the matching release binary, leaving that edit and the new fork request intact.
The retry took 548.8 seconds. Its binary SHA-256 is
`b368a5805fce25d1f35d7be1e940aab5a02dfe625c7adbf9fd67f0fa9a394ab5`.

## Where the endpoint error is

Same saved state as the earlier panel: step 128196, age 4, target 88; CPU/f32,
fixed genome, no runtime references. Values below are at offset 512. Percentages
in the next table are contributions to the sum-of-RMS recovery distance.

| Case | Arm | Micro share | Macro share | Memory share |
|---|---|---:|---:|---:|
| Noise | Off | 18.39% | 81.06% | 0.55% |
| Noise | Fixed diffusion | 17.02% | 77.68% | 5.31% |
| Patch | Off | 12.35% | 80.17% | 7.48% |
| Patch | Fixed diffusion | 20.39% | 67.80% | 11.81% |

A larger share can occur even when absolute error falls. For example, fixed
noise's total normalized distance is only .068936 versus 1.125844 without
diffusion. Memory's larger percentage there is not an increase in memory error.

For patch damage, the absolute field RMS values show what worsened:

| Field | Off | Fixed diffusion | Change |
|---|---:|---:|---:|
| Micro | .00233990 | .00422960 | +80.76% |
| Macro | .01518637 | .01406668 | -7.37% |
| Memory | .00141597 | .00245083 | +73.09% |

Thus, the macro improvement is outweighed by the other two fields. This accounts
for the aggregate ratio rising from .372461 to .407950. It does not by itself
identify which causal coupling produced those changes.

## Which spatial scales remain

The following percentages partition squared-L2 residual energy **within the
macro field**, not the total recovery distance. Low excludes the per-channel DC
component; radial band cutoffs remain .125/.25 cycles per original grid cell.

| Case | Arm | DC | Low | Mid | High |
|---|---|---:|---:|---:|---:|
| Noise | Off | .03% | 5.22% | 15.39% | 79.36% |
| Noise | Fixed diffusion | 6.38% | 72.91% | 19.77% | .94% |
| Patch | Off | 7.39% | 55.39% | 6.57% | 30.65% |
| Patch | Fixed diffusion | 15.08% | 83.90% | .67% | .35% |

DC plus low account for 98.98% of the surviving macro patch energy under diffusion.
It is mainly a spatially varying low-frequency deficit, rather than exclusively
a uniform offset. The micro patch residual under diffusion is also mostly DC plus
low (90.62%). These observations locate residuals; they do not prove a learned
repair mechanism, generalization, or improved perceptual quality.

## Exploratory fork available

The user-provided fork is `/sdcard/Download/titan_image_v9_dynamics_diffusion`,
run tag `v9-dynamics-diffusion-01`. A snapshot taken for this task is at step
130000, age 16, target 38. Its checkpoint is newer than the last completed run
metadata, which describes an interrupted run ending at 128324. The snapshot
passed strict frozen loading and a one-step reference-free development check.

This fork is authorized for exploratory work. It has differentiable RMSNorm and
the saturation penalty; its name does not activate the frozen diffusion sidecar
in training. Its different weights, target, and saved age make comparisons with
the archived panel exploratory rather than a matched learning result.

## Next bounded experiment

On the exploratory fork, compare macro-only, micro-only, and both-grid fixed
diffusion against the undiffused control, retaining the residual and render-detail
measurements. This can test whether macro-only filtering keeps the noise benefit
while avoiding the extra micro/memory patch error and strong rendering changes.
It is a hypothesis to test, not an established remedy. No such additional ablation
or training extension was started in this task.

Evidence:

- `analysis/frozen_residual_recovery_20260913_retry/completed.json`
- `analysis/frozen_residual_recovery_20260913_retry/independent_audit.json`
- `analysis/residual_review_20260913/audit.py`
- `analysis/residual_review_20260913/source_3d7a1aa/` (matching source and binary)
- `analysis/exploratory_fork_130000_20260913/load_check/summary.json`
- [Earlier aggregate recovery and detail results](FROZEN_DIFFUSION_RECOVERY_RESULTS_2026-09-13.md)
