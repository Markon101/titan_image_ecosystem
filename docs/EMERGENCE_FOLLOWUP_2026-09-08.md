# Frozen follow-up tests — 2026-09-08

The normalization A/B remains nearly unchanged on this one target/seed after longer reference-free development. These tests do not establish generalization, a cycle, or homeostasis.

## Scope and controls

- Two frozen evaluations of the pilot control and differentiable-RMSNorm checkpoints at world step 122324; no training.
- Same executable SHA-256 as the pilot; current source differs from that build in `src/engine.rs`. Unit/parity tests check the current working source, while evaluations use the preserved pilot binary.
- Target 0, seed 42, actual 64-step guided burn-in, autonomous horizons 128 and 512, sampling every 32 steps, recurrence history capacity 17.
- Recovery window extended from 32 to 64 steps; six damage cases under guided and autonomous conditions, plus a separate clock-sequence case.
- Same 192px output setting and all other numerical/training settings as each pilot arm. Only output directory and panel settings changed.
- Limits: two evaluations, no agent delegation, no training sweep, 1200-second timeout per evaluation, bounded logs, no base64 image extraction.

## Results

| Metric | Control | Differentiable RMSNorm |
|---|---:|---:|
| Guided target L1 | 0.051187675 | 0.051187679 |
| Autonomous target L1, 128 steps | 0.23567106 | 0.23567103 |
| Autonomous target L1, 512 steps | 0.24206267 | 0.24206264 |
| Image L1 change from withdrawal, 512 steps | 0.25184181 | 0.25184187 |
| Micro state RMS, 512 steps | 1.1830744 | 1.1830744 |
| Macro state RMS, 512 steps | 0.78363764 | 0.7836377 |
| Maximum sampled clamp fraction | 0 | 0 |
| Guided damage cases halving state distance by 64 steps | 4/6 | 4/6 |
| Autonomous damage cases halving state distance by 64 steps | 4/6 | 4/6 |

Halving counts mean the trajectory crossed half its initial state distance at least once. They do not establish sustained recovery. Clock cases begin with identical states and are excluded from those counts.

## Recovery details

| Mode / damage | Control half-time | RMSNorm half-time | Control final/initial distance | RMSNorm final/initial distance |
|---|---:|---:|---:|---:|
| guided / micro_noise | 27 | 27 | 0.298103 | 0.298095 |
| guided / macro_noise | None | None | 0.838087 | 0.838085 |
| guided / micro_patch | 28 | 28 | 0.150445 | 0.15043 |
| guided / macro_patch | None | None | 0.994393 | 0.994359 |
| guided / memory_noise | 2 | 2 | 0.0945005 | 0.0944973 |
| guided / combined | 29 | 29 | 0.38894 | 0.388939 |
| autonomous / micro_noise | 25 | 25 | 0.289659 | 0.289658 |
| autonomous / macro_noise | None | None | 0.824535 | 0.824531 |
| autonomous / micro_patch | 28 | 28 | 0.141855 | 0.141863 |
| autonomous / macro_patch | None | None | 0.99837 | 0.998383 |
| autonomous / memory_noise | 2 | 2 | 0.0912508 | 0.0912865 |
| autonomous / combined | 25 | 25 | 0.383859 | 0.383869 |

## Verification

- CPU: 89 library tests plus 8 regression tests passed.
- OpenCL: the differentiable-normalization full-core renderer VJP parity test passed with the OpenCL test gate enabled. The first overly restrictive test selector ran zero tests; `opencl-parity-executed.log` records the corrected, executed test.
- SHA-256 checks preserved all 37 protected source files and both copied checkpoints. All new archived artifacts were hash-verified and both reports matched their immutable archives.
- All 32 sampled autonomous records are finite, record zero reference fidelity/drive, and report absent runtime references. The panel implementation also asserts zero drive at every autonomous step.
- control: 37 verified artifacts; 4/4 original 32–128-step PNGs reproduced byte-for-byte; largest target-L1 prefix difference 0; evaluation 344.7s.
- rmsnorm: 37 verified artifacts; 4/4 original 32–128-step PNGs reproduced byte-for-byte; largest target-L1 prefix difference 0; evaluation 371.2s.
- Prior pilot records: normalization gradient reachability is 0/1600 in legacy versus 1600/1600 in differentiable mode. All 1600 normalization scale values differ between saved A/B models, but mature guided write saturation remains 100% in both arms. These gradient measurements come from the pilot, not new training.
- Direct image inspection was attempted with `view_image` and failed with `filesystem sandbox cannot be enforced on this executor`. No base64 extraction or visual-quality claim was made. PNGs remain in the immutable archives.
- Timings are operational receipts, not isolated performance benchmarks; unit/parity tests overlapped part of the first evaluation.

## Additional observations

- control: closest sampled full-state recurrence at the 512-step endpoint has RMS distance 0.0536471 at lag 64; image correlation is 0.99926. High image correlation alone does not establish a cycle.
  - Changed clock sequence (guided): final state distance 0.124072; rendered L1 difference 0.00332044, from identical initial states.
  - Changed clock sequence (autonomous): final state distance 0.124225; rendered L1 difference 0.00187931, from identical initial states.
- rmsnorm: closest sampled full-state recurrence at the 512-step endpoint has RMS distance 0.0536471 at lag 64; image correlation is 0.99926. High image correlation alone does not establish a cycle.
  - Changed clock sequence (guided): final state distance 0.124072; rendered L1 difference 0.00332044, from identical initial states.
  - Changed clock sequence (autonomous): final state distance 0.124225; rendered L1 difference 0.00187931, from identical initial states.

## Ideal next steps

1. Prioritize longer macro-field recovery observations at 256 and 512 steps, then repeat across more targets/seeds. The current panel runs every damage case; a selective case option would keep this follow-up economical.
2. For longer training, use the prepared 4096-step matched continuations below as a bounded experiment. Check gradients, write saturation, guided reconstruction, and recovery before adding another block; these runs have not started.
3. If saturation persists, test a separate opt-in saturation intervention with a matched control before committing to substantially longer training. Extend general autonomous observation to 2048 steps after these focused checks.

Machine-readable detail: [summary.json](../analysis/followup_2026-09-08/summary.json). Reproduction: `run.py` records the bounded execution, copied inputs, and hashes; `summarize.py` derives this report. The run directory is intentionally single-use. All logs, request files, hash receipts, and scripts are in `analysis/followup_2026-09-08/`.

## Recommended code changes

1. **Selective recovery cases first.** Add an optional recovery-case list to `PanelConfig` in `src/experiment.rs`, consumed by `src/analysis/experimental.rs`. Preserve the existing six-case default. This lets longer 256/512-step runs target macro noise and macro patch damage without paying for every recovery pair. Record selected cases and effective macro update counts in each report.
2. **A separate saturation experiment.** Add an opt-in penalty on excessive pre-tanh write logits, with its configuration included in the training signature and fork provenance. Keep its default weight zero. The existing normalization fix reaches all 1600 normalization parameters, but this pilot still reports 100% mature guided write saturation and nearly unchanged 512-step behavior. A direct logit penalty is a hypothesis to test, not a demonstrated remedy. Verify finite, nonzero gradients through the penalty, reduced saturation, retained guided reconstruction, and matched-budget recovery before adoption.

Keep the differentiable RMSNorm option and legacy baseline. The current evidence does not justify changing default gains, learning rate, BPTT, capacity, or the training objective for existing checkpoints. No implementation changes were made during this follow-up; only tests, analysis artifacts, documentation, and prepared continuation requests were added.

## How to continue the normalization fork

Both pilot arms trained for 64 development steps, including eight full-core updates. Neither has completed a long continuation. The frozen follow-up evaluations perform no training.

The prepared requests create separate descendants of the completed pilot checkpoints, retaining weights, optimizer moments/update count, world state, and warmup position. Each requests 4096 additional development steps, or 512 full-core updates at BPTT 4 and cadence 2. Checkpoints and snapshots are saved every 256 steps. The normalization modes and original training objectives are retained; no withdrawal curriculum or saturation intervention is added.

Run after the frozen evaluations finish:

```sh
cd ~/projects/titan_image_ecosystem
export OCL_ICD_ASSUME_ICD_EXTENSION=1

./target/release/titan_image fork \
  analysis/followup_2026-09-08/long-rmsnorm-request.json &&
./target/release/titan_image --config-json \
  analysis/emergence_2026-09-08_long_rmsnorm/config.json
```

For a later resume, run only the second command. Each invocation requests another 4096 steps from its loaded checkpoint. The fork command intentionally refuses an existing destination.

For a matched legacy control, run sequentially:

```sh
./target/release/titan_image fork \
  analysis/followup_2026-09-08/long-control-request.json &&
./target/release/titan_image --config-json \
  analysis/emergence_2026-09-08_long_control/config.json
```

The request files have been prepared, but neither import nor long training has been executed. A longer run is a measured experiment, not a defined state of being fully trained. Compare saved gradients, write saturation, guided reconstruction, and frozen recovery before extending again.

## Subsequent implementation

Recovery-case selection and the opt-in saturation penalty are now implemented. See [configuration, semantics, and validation](FOCUSED_RECOVERY_AND_SATURATION.md). The experimental results above predate these code changes.
