# Safe fresh normalization experiments

The executable can publish an untrained checkpoint, sample write diagnostics during training, and exclude competing writers to a run. Defaults retain legacy normalization and disable write diagnostics. These additions do not alter learned-evolution signatures or normal checkpoint validation.

## Run ownership and invocation identity

Training and analysis acquire an OS advisory lock on `.titan_image_writer{suffix}.lock` before reading or changing the run's checkpoint. Fork/import and initialization acquire the same ownership guard in their new destination. Different run tags normally have separate artifact namespaces and locks. On filesystems that reject file locks (including Android shared storage returning ENOSYS/EOPNOTSUPP), ownership falls back to an exclusive lock on the existing output directory. That fallback serializes all run tags in the directory. If directory locking also fails, the operation is rejected; unsupported locking never permits an unguarded writer. The ownership receipt records `lock_scope` as `run_tag` or `output_directory`. Read-only parent import still uses strict checkpoint loading and before/after hashes.

A competing writer fails immediately. Normal lease teardown explicitly unlocks before closing, so a descriptor briefly inherited by a subprocess cannot block an immediate resume. Process exit also releases ownership once its open file descriptions close; the ownership file stays in place and is reused. Do not delete that file to override a running process: replacing its inode would defeat exclusion. The guard coordinates executables containing this change; earlier binaries do not participate.

Each invocation writes a separate `titan_image_invocation{suffix}_{id}.json` startup receipt. Its ID is included in run/render metadata, optimizer diagnostics and write diagnostics. Legacy `run_id` and CSV columns remain available. Group new diagnostic rows by `invocation_id`; `world_step` alone is not a unique key across resumed or repeated runs.

## Periodic write diagnostics

Add `--write-diagnostics-every 32` to a training command, or set `experiment.write_diagnostics_every` in JSON. Zero is the default and disables sampling. Within each invocation, N samples immediately before full-core windows 1, 1+N, 1+2N, and so on; decoder-only windows are skipped. It can be enabled on an existing checkpoint without a fork.

`write_diagnostics{suffix}.jsonl` records invocation, target, episode, age, world step, optimizer update, fidelity and diagnostic timing. Each sample uses one untracked CPU interface forward pass on the current world and weights. It does not advance the world, perturb inputs, accumulate gradients, change weights or run the expensive multi-input sensitivity panel. It reports:

- Micro and macro pre-tanh logit distributions, saturation fraction, derivative distributions and spatial write variance.
- Token RMS through the interface loops.
- The current reference fidelity, including withdrawal scheduling when configured.

This measures interface conditioning; it does not include a local reference-drive dynamics update. Write saturation is distinct from state clamp fractions and is a diagnostic, not a success criterion or a new loss. The saturation penalty remains off unless explicitly configured separately.

## Initialize without training

The `initialize` subcommand accepts the normal configuration flags and requires a new output directory:

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 ./target/release/titan_image initialize \
  --corpus-dir /path/to/training/images \
  --output-dir /path/to/new/initialization \
  --run-tag initial --profile s25-fast --style pure-nca \
  --compute-backend opencl --training-rmsnorm legacy
```

It applies the same deterministic parameter and initial-world construction as normal fresh training, then saves and strictly reloads the model/world/optimizer checkpoint at step zero. Initialization itself runs on CPU and applies zero optimizer updates. It writes `config.json` with `fresh: false`, so subsequent invocations resume safely. Failed initialization leaves `.fork-incomplete`; ordinary resume refuses that directory. Existing destinations are rejected, even with `--fresh`.

## Prepare an identical fresh A/B pair

```sh
python scripts/fresh_norm_pair.py \
  --corpus-dir /path/to/training/images \
  --root analysis/fresh_norm_seed42 \
  --seed 42 --core-updates 128
```

The helper defaults to `s25-fast`, `pure-nca`, OpenCL, eight threads, and diagnostics every 32 full-core windows. It creates one untrained initialization and imports two descendants: `legacy` and `rmsnorm`. Both retain initial world, optimizer and warmup state. Both have the saturation penalty disabled. It verifies identical tensor names, shapes, dtypes and bytes in model, world and optimizer, excluding only fork-specific identity scalars, and verifies that the initial checkpoint files remain unchanged.

Preparation launches no training. Use the pair only after `completed.json` exists and `.pair-incomplete` is absent. Identity and initialization checks remain active under `python -O`; a failed check leaves the pair incomplete. Its receipt records exact commands, initial hashes, binary hash, arm hashes and the matched update budget. With the current fast profile, 128 full-core updates from step zero require 1020 development steps: the first window is a core update, and the final requested core window need not be followed by a decoder-only window.

Start the arms sequentially with the saved configuration files:

```sh
OCL_ICD_ASSUME_ICD_EXTENSION=1 RAYON_NUM_THREADS=8 \
  ./target/release/titan_image --config-json analysis/fresh_norm_seed42/legacy/config.json

OCL_ICD_ASSUME_ICD_EXTENSION=1 RAYON_NUM_THREADS=8 \
  ./target/release/titan_image --config-json analysis/fresh_norm_seed42/rmsnorm/config.json
```

Every invocation requests its configured number of **additional development steps**. The matched full-core count in the preparation receipt applies to the initial run from step zero. Recompute a later continuation budget from its loaded step/cadence; do not assume repeating the same development-step budget gives the same full-core count.

Compare reconstruction, write responsiveness and macro recovery together before extending either arm. This infrastructure does not establish a learning benefit for a fresh model, and no fresh research training is launched during implementation validation.
