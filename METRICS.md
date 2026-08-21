# TITAN Image v7 metrics and experiment protocol

v7 writes one CSV row per optimizer window. No scalar proves image quality,
learning, novelty, generalization, or an attractor; inspect raw outputs, state
atlases, trajectories, and matched controls.

## New v7 evidence

- reference_fidelity: actual reference strength used for that window. It is zero
  after hybrid conditioning dropout.
- interface_memory_rms: RMS of persisted GRU/MorphicStack global state.
- muon_variables: matrices updated through Muon. It must be zero under AdamW
  and on decoder-only windows that do not reach the interface.

Existing fields retain their meanings: scale-separated movement, state RMS and
near-clamp occupancy, image-space motion, color/structure diagnostics, loss
components, reached parameters, gradient RMS/clipping, learning rate, and
window time.

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

v7 still needs a family-disjoint manifest, fixed development/validation probes,
nearest-training-image distance, seed-diversity distance, and fidelity-stratum
summaries. Until those exist, reconstruction and gallery results are
developmental rather than held-out generalization evidence.
