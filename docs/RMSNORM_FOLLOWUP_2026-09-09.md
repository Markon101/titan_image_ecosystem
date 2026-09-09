# RMSNorm fork follow-up — September 9, 2026

Status: probe comparisons and CPU gradient checks complete; 512-step recovery evaluations running. A sequential 32-full-core-update matched saturation/control pilot and final report/commit are queued. This file will be replaced with the complete verified results by `scripts/finish_frozen_followup.py`.

Parent checkpoint: 122260. Latest fork: 128196. Both frozen evaluations use OpenCL, eight threads and 192px output. Original checkpoints are read-only inputs; copied checkpoints retain their hashes after each evaluation.

| Age | Reference fidelity | Parent mean RGB L1 | Fork mean RGB L1 |
|---:|---:|---:|---:|
| 8 | 1 | 0.412398 | 0.285630 |
| 8 | 0 | 0.311113 | 0.356471 |
| 64 | 1 | 0.104162 | 0.061026 |
| 64 | 0 | 0.440101 | 0.430124 |

Six identical held-out source images were compared. Age-64 guided error decreased on all six, averaging 41.41% lower. Early zero-reference error worsened. These measurements do not establish autonomous persistence or damage recovery.

CPU probes reached 0/10 normalization tensors in the parent and 10/10 (1600 scales) in the fork; all computed gradients were finite and no optimizer updates were applied. Losses are not directly comparable because these probes start from different saved worlds and objectives.

The latest training invocation contains 309 contiguous optimizer windows, including 155 full-core updates. All 1600 normalization parameters were reachable in each full-core window and the saturation penalty was active. The full historical log has 289 non-increasing step transitions and is unsuitable as one continuous learning curve.

The experiment root is `analysis/rmsnorm_followup_2026-09-09_matched`. It records exact configurations, source and binary hashes, per-job receipts and immutable artifacts. An earlier mixed-backend attempt was stopped and excluded; protected files remained unchanged. The image viewer failed with the Termux filesystem sandbox error, so visual quality was not assessed.

The finalizer waits for both experiments, verifies their evidence, replaces this report, saves `docs/evidence/rmsnorm_followup_2026-09-09.json`, packages `analysis/rmsnorm_followup_2026-09-09_matched/report.html`, and creates a second local Git commit. It fails closed if either experiment fails. It never pushes.

Monitor:

```sh
tail -5 /data/data/com.termux/files/usr/tmp/rmsnorm-followup-matched.log
tail -5 /data/data/com.termux/files/usr/tmp/rmsnorm-saturation-queue.log
tail -5 /data/data/com.termux/files/usr/tmp/rmsnorm-finalizer.log
```

Do not restart the single-use runners in their existing directories. A successful finish writes `analysis/rmsnorm_followup_2026-09-09_matched/finalized.json` with the final commit identity. Until then, recovery and penalty-effect conclusions remain pending.
