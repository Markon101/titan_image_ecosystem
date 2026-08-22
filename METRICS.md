# TITAN Image v8 metrics and experiment protocol

v8 writes one CSV row per optimizer window. No scalar proves image quality,
learning, novelty, generalization, or an attractor; inspect raw outputs, state
atlases, trajectories, and matched controls.

## New v8 evidence

- `reference_fidelity`: actual reference strength used for that window. It is
  zero after hybrid conditioning dropout.
- `interface_memory_rms`: RMS of persisted bounded GRU/MorphicStack memory.
- `loss_state` and `loss_memory`: unweighted soft-barrier energies. Multiply
  them by the configured weights when comparing their contribution to total
  loss.
- `core_gradient_rms` and `decoder_gradient_rms`: separately normalized
  gradient energy. Decoder-only rows correctly report zero core parameters and
  zero core gradient RMS.
- `core_updated_parameters` and `decoder_updated_parameters`: make a dead
  core distinguishable from a renderer-only update.
- `stability_violation`: the current row exceeded the configured near-bound
  occupancy threshold.
- `development_steps_per_second`: development steps divided by full optimizer
  window wall time, including forward, render/loss, backward, optimizer, and
  diagnostics.
- `muon_variables`: matrices updated through Muon. It is zero under AdamW and
  on decoder-only windows.

`image_variance` is now the mean of the three within-channel spatial
variances. A spatially uniform cyan frame therefore reports zero rather than a
false high value caused only by differences between RGB channel means.
`micro_clamp_fraction` and `macro_clamp_fraction` retain their CSV names for
continuity but now mean occupancy above 99% of the smooth state bound; there is
no hard state clamp in v8.

## Conditioning interpretation

Compare like with like. Reconstruction windows have easier input than
null-reference generation windows. Report losses grouped by fidelity:

- null: f = 0;
- loose: 0 < f < 0.4;
- balanced: 0.4 <= f < 0.75;
- faithful: f >= 0.75.

A declining reconstruction loss does not prove autonomous generation improved.
Hybrid training succeeds only when null-reference quality, conditioned
reconstruction, and seed diversity remain viable.

## Recurrent-interface interpretation

The interface write heads start at zero, so early runs initially preserve the
parent local architecture. Useful-interface evidence includes reached
variables, bounded memory RMS, lower clamp reliance, fixed-probe improvement,
and global changes that palette matching alone cannot reproduce.

Higher memory RMS, more loops, or more morph blocks is not evidence of better
global reasoning.

## Stability interpretation

A healthy run keeps memory below `memory_limit`, near-bound occupancy low,
core gradient RMS nonzero on full-core windows, and image spatial variance and
edge energy above numerical zero. A renderer can still lower image loss while
the organism is frozen, so output loss alone is not sufficient.

After `stability_patience` consecutive violating windows, training stops at a
window boundary, flushes metrics, saves model/world/optimizer/manifest, writes
final diagnostics, and skips additional gallery development. The checkpoint is
structurally resumable, but continuing unchanged is normally inappropriate;
inspect it with `--render-only` or start a corrected fresh tag.

## Muon protocol

A Muon claim requires matched AdamW and hybrid-Muon runs with identical corpus,
seed, architecture, conditioning schedule, augmentation, update count, and
thermal policy. Compare quality per wall-clock hour, optimizer and total time,
clipping/nonfinite updates, fidelity strata, and at least three seeds.

## Minimum architecture ablations

1. Interface disabled with --interface-gain 0.
2. One versus the profile loop count.
3. --morph-depth 1 versus the profile default.
4. generate, hybrid, and reconstruct conditioning.
5. AdamW versus hybrid Muon.
6. Physical-operator ablations under identical interface settings.

Changing a style preset changes multiple controls and is not a single-variable
ablation.

## Evidence still missing

v8 still needs a family-disjoint manifest, fixed development/validation probes,
nearest-training-image distance, seed-diversity distance, and fidelity-stratum
summaries. Until those exist, reconstruction and gallery results are
developmental rather than held-out generalization evidence.
