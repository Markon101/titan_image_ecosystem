# RMSNorm fork follow-up — September 9, 2026

## Technical summary

Completed frozen parent/fork comparisons and a separate 32-full-core-update saturation/control pilot. The longer fork combines additional training, differentiable RMSNorm and a write-logit penalty; its before/after differences cannot isolate any one cause. The short pilot tests the penalty at a matched budget. No broad training extension was launched. Use the per-case results below to decide the next bounded experiment.

## Scope and definitions

Parent step 122260; latest fork step 128196. Both frozen evaluations use OpenCL, 192px output and eight threads; gradient probes use CPU. Six held-out images are measured at ages 8 and 64, with full reference or no runtime references. Two familiar targets (0, 41) and two seeds (42, 137) receive 64 guided development steps and 512 autonomous steps. Each macro damage case is paired with an undamaged trajectory using the same clock sequence. Raw L1 is mean absolute RGB error; lower means closer reconstruction, not necessarily better autonomous dynamics. Write saturation is the fraction of absolute pre-tanh logits above 2.6466525. Recovery distance is the sum of micro, macro and memory RMS differences, divided by its initial value. The final-interval criterion checks all saved samples from steps 384 through 512, not every intermediate step.

## Held-out reconstruction

| Age | Fidelity | Parent mean L1 | Fork mean L1 | Fork change |
|---:|---:|---:|---:|---:|
| 8 | 1 | 0.412398 | 0.285630 | -30.74% |
| 8 | 0 | 0.311113 | 0.356471 | +14.58% |
| 64 | 1 | 0.104162 | 0.061026 | -41.41% |
| 64 | 0 | 0.440101 | 0.430124 | -2.27% |

Age-64 guided error improved on all six probes. The age-8 zero-reference mean worsened; the benefit is not uniform across reference conditions and ages.

## Familiar target retention

| Target | Seed | Parent guided L1 | Fork guided L1 |
|---:|---:|---:|---:|
| 0 | 42 | 0.080574 | 0.058058 |
| 0 | 137 | 0.080368 | 0.057838 |
| 41 | 42 | 0.130216 | 0.104439 |
| 41 | 137 | 0.130328 | 0.104414 |

These familiar targets check retention on a small panel; they do not estimate whole-corpus performance.

## Write saturation and gradients

| Checkpoint | Head | Mean mature guided saturation | Mean tanh derivative |
|---|---|---:|---:|
| parent | micro | 1.000000 | 2.30293e-08 |
| parent | macro_field | 1.000000 | 9.52402e-09 |
| fork | micro | 0.984375 | 0.0029423 |
| fork | macro_field | 0.674561 | 0.0252593 |

The CPU gradient probes reached 0/10 normalization tensors in the parent and 10/10 (1600 scales) in the fork, with no optimizer updates. Saturation and derivative measurements above assess write responsiveness separately from state clamps.

## Macro recovery

| Checkpoint | Mode | Damage | Median ratio 256 | Median ratio 512 | Final sampled interval below half |
|---|---|---|---:|---:|---:|
| parent | guided | macro_noise | 0.446925 | 0.206243 | 4/4 |
| parent | guided | macro_patch | 0.355453 | 0.185626 | 4/4 |
| parent | autonomous | macro_noise | 0.431062 | 0.201931 | 4/4 |
| parent | autonomous | macro_patch | 0.318327 | 0.162536 | 4/4 |
| fork | guided | macro_noise | 1.684146 | 2.675765 | 0/4 |
| fork | guided | macro_patch | 0.746387 | 0.725612 | 0/4 |
| fork | autonomous | macro_noise | 0.698091 | 0.934200 | 0/4 |
| fork | autonomous | macro_patch | 0.720803 | 0.651471 | 0/4 |

Each 512-step recovery contains 128 macro updates. Ratios below one indicate contraction relative to the initial damage; threshold crossings alone do not establish sustained recovery or homeostasis. Exact per-pair endpoints and trajectories are retained in the evidence.

## Matched saturation intervention

| Arm | Guided L1 | Mature micro saturation | Mature macro saturation | Recovery cases ending below half |
|---|---:|---:|---:|---:|
| control | 0.056313 | 1.000000 | 1.000000 | 4/4 |
| penalty | 0.089920 | 0.593750 | 0.526855 | 0/4 |

Both arms start from identical model, world and optimizer tensor values, excluding only fork-specific identity scalars. Both enable differentiable RMSNorm and retain the optimizer and warmup position. Only the penalty differs (zero versus weight 0.0001, threshold 2.5). Each trains 256 development steps / 32 full-core updates. This is a one-target, one-seed pilot, not a causal explanation of the longer fork.

## Validation and limitations

All frozen checkpoint copies, original protected files and probe sources retained their hashes. All archived artifact hashes were checked. Autonomous samples record zero reference drive and absent runtime references. The binary SHA-256 matches the earlier CPU/OpenCL-validated build and its 37-file source snapshot; no Rust numerical code changed in this task. The inherited backend mismatch was detected in an initial attempt, which was stopped and excluded before rerunning both arms on OpenCL. The training log contains 289 non-increasing step transitions; only its latest contiguous 309-window invocation was used for mechanical checks. Image inspection failed with the Termux filesystem sandbox error, so no visual-quality claim is made. HTML verification is structural only when Chromium is unavailable.

## Decision and further questions

Do not launch an unrestricted continuation based on guided reconstruction alone. The next step must address the weakest measured condition in the tables: retention if familiar errors regress, write responsiveness if saturation remains high, or autonomous macro recovery if damage does not contract. A favorable one-seed penalty result warrants replication across the two-target/two-seed panel before another long block. Keep RMSNorm, penalty, withdrawal, BPTT and replay changes isolated in explicit forks; this study does not establish cycles or homeostasis.

## Reproduction

Run `scripts/frozen_fork_followup.py --help` and `scripts/matched_saturation_pilot.py --help` for the single-use runners. Each analysis root retains the exact plan, configurations, command receipts, hashes and immutable evaluation archives. Run `python scripts/summarize_frozen_fork.py analysis/rmsnorm_followup_2026-09-09_matched` to reverify the frozen evidence. [Full compact evidence](evidence/rmsnorm_followup_2026-09-09.json) contains every reported comparison and archive identity.

HTML packaging failed; the validated Markdown write-up and JSON evidence remain available. See report-delivery.json.
