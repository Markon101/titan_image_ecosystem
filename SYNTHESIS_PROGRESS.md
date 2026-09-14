# Synthesis implementation checkpoint

## Piece 1: per-grid frozen diffusion — source complete

`src/development/operators.rs` now accepts `diffusion_scope`:
`both` (default), `micro`, or `macro`. A standalone frozen sidecar can contain:

```json
{"diffusion_max": 0.02, "diffusion_scope": "macro"}
```

Effective constant nu is .01 when logits are zero. Scope changes only diffusion;
transport and macro cadence are preserved. No checkpoint or training changes.
Validation: 17 focused Rust tests, strict all-target OpenCL-feature Clippy, and
format/diff checks passed. Tests include signed-zero preservation on excluded
grids, analytical stencil output, legacy defaults, transport, and invalid scope.

Build status: check `analysis/synthesis_piece1_20260913/release_receipt.json`.
If absent, the release rebuild remains pending; source changes alone do not make
the existing executable support the new field. Build logs are in that directory.
The build uses a committed source snapshot to exclude unrelated workspace edits.

## Exact next piece: expose scope in recovery

1. Add an optional recovery CLI enum, e.g. `--recovery-diffusion-scope`, default
   `both`. Reject its use outside recovery; preserve existing default results.
2. Construct the fixed arm and its manifest from one shared operator definition,
   avoiding separate hardcoded copies. Include the selected scope in the protocol.
3. Wire selection through `scripts/frozen_diffusion_recovery.py`, keeping each
   comparison in a new output root. Add focused CLI and manifest tests.
4. Rebuild the analysis release executable and run a tiny smoke before a long run.

The recovery CLI does NOT expose scope yet. `--operators FILE` in ordinary frozen
mode can use the new sidecar once the new executable has been built.

## Following pieces

- Implement explicit genome inputs and matched fresh-state synthesis sampling.
- Add native/mixed/mutated genome panels with immutable receipts and raw galleries.
- Add diversity, conditioning-response, detail, temporal, and offline corpus checks.
- Compare off/micro-only/macro-only/both on a small subset; extend useful cases.
- Consider short reference-withdrawal training only after synthesis evaluation.

See [SYNTHESIS_PLAN.md](SYNTHESIS_PLAN.md) for scope and acceptance criteria.
The authorized exploratory fork is `/sdcard/Download/titan_image_v9_dynamics_diffusion`;
recheck its current checkpoint and active processes before using it. The verified
snapshot was step 130000, age 16, target 38.

Preserve unrelated `src/run_lease.rs`, `fork_diffusion_request.json`, local render
helpers, and bytecode. No training or empirical synthesis panel was started here.
Keep each subsequent piece independently tested and committed with an updated
checkpoint here, so quota exhaustion does not strand an ambiguous partial change.
