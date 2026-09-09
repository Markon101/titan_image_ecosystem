# RMSNorm fork follow-up — September 9, 2026

## Technical summary

Do not extend the current penalty-enabled fork yet. Guided reconstruction improved on all six held-out probes and all four familiar target/seed pairs, but macro recovery regressed: all 16 parent recovery cases stayed below half their initial distance in the final sampled interval, versus 0/16 for the latest fork. In the matched 32-full-core-update pilot, the penalty reduced write saturation but increased guided error by 59.68% and worsened recovery in every tested condition. The training-extension gate failed. No broad continuation was launched.

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

Guided error improved in all four familiar target/seed pairs. After 512 reference-free steps, mean target L1 also decreased: 0.239736 to 0.225255 for target 0 and 0.377822 to 0.348753 for target 41. These are reconstruction measurements on a small panel, not evidence of whole-corpus performance or damage recovery.

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

The longer horizon changes the earlier diagnosis: the saturated parent recovers slowly, whereas the latest fork retains much more damage. Guided macro-noise distance grows to a median 2.676 times its initial value in the latest fork. Each 512-step recovery contains 128 macro updates. Ratios below one indicate contraction relative to the initial damage; threshold crossings alone do not establish sustained recovery or homeostasis. Exact per-pair endpoints and trajectories are retained in the evidence.

## Matched saturation intervention

| Arm | Guided L1 | Mature micro saturation | Mature macro saturation | Recovery cases ending below half |
|---|---:|---:|---:|---:|
| control | 0.056313 | 1.000000 | 1.000000 | 4/4 |
| penalty | 0.089920 | 0.593750 | 0.526855 | 0/4 |

Both arms start from identical model, world and optimizer tensor values, excluding only fork-specific identity scalars. Both enable differentiable RMSNorm and retain the optimizer and warmup position. Only the penalty differs (zero versus weight 0.0001, threshold 2.5). Each trains 256 development steps / 32 full-core updates. The penalty arm has 59.68% higher guided L1 than the control. Its guided macro-noise and macro-patch distances finish at 9.313 and 3.167 times their initial values, versus 0.203 and 0.158 for control. Its autonomous ratios are 0.678 and 0.802, versus 0.203 and 0.153. This is a one-target, one-seed pilot, not a causal explanation of the longer fork.

## Validation and limitations

All frozen checkpoint copies, original protected files and probe sources retained their hashes. All archived artifact hashes were checked. Autonomous samples record zero reference drive and absent runtime references. The binary SHA-256 matches the earlier CPU/OpenCL-validated build and its 32-file Rust source snapshot; no Rust numerical code changed in this task. The inherited backend mismatch was detected in an initial attempt, which was stopped and excluded before rerunning both arms on OpenCL. The training log contains 289 non-increasing step transitions; only its latest contiguous 309-window invocation was used for mechanical checks. Image inspection failed with the Termux filesystem sandbox error, so no visual-quality claim is made. The HTML report passed payload and structural validation. No compatible Chromium was installed, so interactive rendering, source dialogs and viewport layout were not browser-tested.

## Decision and further questions

The current penalty setting (weight 0.0001, threshold 2.5) fails the measured extension gate. Preserve the latest fork as evidence, but do not add a long training block or disable the penalty in place. Retain strict fork signatures and the original checkpoint.

The next controlled experiment should use the RMSNorm-only control checkpoint as its starting point and test a weaker penalty in a separate matched fork. Keep the budget short and require retained guided reconstruction plus improved or preserved macro recovery before extending. A weaker weight is a hypothesis, not an approved remedy. No such extra experiment was run here.

Lower saturation alone is not a success criterion: this intervention made writes more responsive while trajectories became less robust. Conversely, the parent’s strong distance contraction may reflect insensitive saturated dynamics; it does not by itself prove useful learned repair. Separate perturbation robustness, target retention and responsiveness when choosing the next objective. Withdrawal, BPTT and replay changes should remain separate experiments. Neither result establishes a cycle or homeostasis.

## Reproduction

Run `scripts/frozen_fork_followup.py --help` and `scripts/matched_saturation_pilot.py --help` for the single-use runners. Each analysis root retains the exact plan, configurations, command receipts, hashes and immutable evaluation archives. Run `python scripts/summarize_frozen_fork.py analysis/rmsnorm_followup_2026-09-09_matched` to reverify the frozen evidence. [Full compact evidence](evidence/rmsnorm_followup_2026-09-09.json) contains every reported comparison and archive identity.
