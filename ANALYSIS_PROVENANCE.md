# Analysis artifact ownership (provenance version 3)

A completed evaluation owns only outputs emitted through its writer, in its own
new directory. File existence is not evidence of ownership. There is no output
cache or implicit reuse. Checkpoint schema remains v9; autonomous record version
remains 2 because its numerical definition did not change.

## Failure investigation

The September 7 archive `step_121644_1788767201690678250-11028-0.json` claimed 35
root-level artifacts. On-device inspection found all 35 had earlier mtimes
(August 31 through September 4), including the reported September 2 probe PNGs.
The archive recorded file sizes and FNV hashes but no evidence that the evaluation
had written those files.

`engine::run` disables training through `training_enabled = !render_only &&
!analysis.only`, loads the checkpoint, renders final outputs, then invokes
`analysis::run_checkpoint_analysis`. Probe trajectories are computed in
`probe::run_natural_image_probes`: each requested age renders a fresh tensor,
calls the PNG writer, and computes reconstruction/image/state metrics from that
same tensor. The PNG helper `render::save_planar_png` encodes a temporary PNG
and renames it over the destination, propagating errors. Neither it nor the
probe runner has an existence-based skip. Montage writers likewise encode and
rename unconditionally for nonempty inputs.

The ownership bug was in the shared architecture: filenames contained run tag,
target fingerprint, rounded fidelity and age, but no evaluation ID. Other
families also reused canonical filenames. `provenance::completed_artifacts`
reconstructed expected paths after analysis and hashed whatever bytes it found.
Successful replacement could invalidate earlier archives; stale files could be
adopted without detection. Individual resolution and target-comparison images
were missing from the inventory altogether. There is no implemented output-cache
policy: the defect is accepting unproven ownership and mutable archive references.

The timestamp-based diagnosis needed correction. In a disposable `/sdcard`
reproduction, a successful temporary-PNG rename showed a fresh mtime immediately,
but a separate tool call reported the old destination mtime. This happened for
both identical-byte replacement and changed-byte replacement; the changed bytes
persisted even though the mtime appeared old again. A Termux temporary-directory
control retained the fresh mtime across calls. This establishes an observed
shared-storage metadata inconsistency across execution calls, not a renderer
existence check or proof that bytes were not rewritten. The underlying storage
or execution-layer implementation responsible for that inconsistency is not
identified here.

Freshly regenerated outputs further constrain the diagnosis: 34 of the 35 old
artifacts are byte-identical to their new evaluation-owned counterparts. The
remaining probe report differs only in its output paths; all probe metrics are
identical. Thus incorrect probe pixel contents were not demonstrated for this
checkpoint/settings. The architecture fix is still necessary: old timestamps,
matching hashes, and completion flags cannot establish who generated a file, and
mutable canonical paths cannot retain trustworthy historical evaluation outputs.

## Implementation

| Module/functions | Change |
| --- | --- |
| `analysis/artifacts.rs`: `EvaluationArtifacts::{new,write,png,montage,json,publish}` | Reserve a new evaluation directory, reject occupied output paths, register bytes only after successful generation, validate montage inputs and report references against write receipts, recheck identities before publication, consume the writer at publication. |
| `analysis/provenance.rs`: `AnalysisProvenance::new`, `file_identity` | Provenance version 3, evaluation-owned archive path and explicit fresh-output policy; delete the post-hoc filename collector. Checkpoint identities remain separate inputs. |
| `analysis.rs`: `run_checkpoint_analysis` and artifact-producing helpers | Return actual emitted paths and register decomposition/frontier, complete resolution ladder, target comparison, autonomous, perturbation, flow and benchmark outputs. Publish only after the engine's other evaluation outputs succeed. |
| `probe.rs`: `run_natural_image_probes` | Write each requested render, montage and report through the same writer. Include exact fidelity bits in filenames to distinguish values such as 0.5 and 0.5001 that both round to f0500. Numerical trajectory and metric computation are unchanged. |
| `engine.rs`: `run`, `write_evaluation_output` | Include analysis-only final renders, mastered/grounded/emergent images, state atlases, model statistics and render metadata in the evaluation. Preserve normal training output behavior. |
| `comparison.rs`: `compare_v8_v9`, `snapshot` | Use the current in-memory summary instead of reopening a potentially stale canonical analysis JSON. Emit the comparison through the evaluation writer and identify its source archive; saved v8/v9 training metrics remain explicitly named historical inputs. |

Layout preserves descriptive basenames in one flat, evaluation-specific directory:

```text
analysis_history_v9_<tag>/
  step_<world-step>_<nanoseconds>-<pid>-<sequence>/
    evaluation.json
    titan_image_probe_...png
    titan_image_probe_report_...json
    titan_image_attractor_analysis_...png
    ...
```

`create_dir` rejects evaluation-ID collisions. Output reservations reject
accidental repeated destinations and pre-existing files. A failed or empty
write prevents publication; partial directories without `evaluation.json` are
incomplete. The program never reopens a completed evaluation for writing.
This is application-level immutability, not protection against external editing;
FNV-1a and lengths detect accidental changes, not adversarial tampering.

The canonical analysis JSON and render metadata remain latest conveniences.
Consumers must follow recorded paths for other analysis sidecars and images.
Old archives/canonical artifacts are retained untouched and cannot retroactively
acquire version-3 guarantees. Source-pyramid caching remains an input optimization
and is never registered as evaluation output. Training/checkpoint formats,
optimizer behavior, and autonomous zero-reference equations are unchanged.

## Reproducible regressions

```sh
cargo test --locked --all-targets --features opencl
cargo clippy --locked --all-targets --features opencl -- -D warnings
cargo fmt --check
cargo build --release --locked --features opencl -j 8
```

`analysis::regression::fresh_evaluations_never_adopt_canonical_or_prior_outputs`
pre-creates invalid stale PNGs/reports at legacy and current canonical templates,
runs two real small-model evaluations, validates every emitted byte length/hash,
decodes PNGs, checks PNG channel means against fresh probe metrics to 8-bit
quantization tolerance, verifies exact report serialization, and checks that the
first archive and all its artifacts survive the second evaluation unchanged.
It exercises probes, fidelity collisions, decomposition/frontier, resolution
images and crop, target images, autonomous frames/montages, perturbation,
benchmark, and flow. Forged canonical references and post-write tampering cannot
produce a completed archive.

`analysis::artifacts::tests::writer_rejects_noops_collisions_and_unowned_montage_inputs`
checks empty generation, existing evaluation destinations, repeated writes and
foreign montage inputs. Existing engine fresh/resume/probe tests check that
model, world, optimizer, manifest, CSV and training metadata bytes stay unchanged.
Existing autonomous tests check exact independence from reference tensors and
preservation of the saved world.

`comparison::tests::comparison_does_not_load_a_stale_canonical_analysis` plants
a stale separability value of 999 in the canonical summary and verifies that a
comparison without a current evaluation cannot import it.

## Android/Termux verification, September 7, 2026

The release build used `--features opencl`, native AArch64 CPU tuning and `+fp16`.
The real invocation was replayed from the existing render metadata, including
run tag `v9-emergent-tuned-c282-01`, world step `121644`, original training corpus
`/sdcard/Download/titan_image_sources_nca256/`, and probe directory
`/sdcard/Download/titan_field_substrate_v3_1_set01`. Probe ages were `64,128`,
fidelity `0.5`, and autonomous horizon/stride `128/32`.

Verified archive, relative to `/sdcard/Download/titan_image_v9_pure_nca/`:

```text
analysis_history_v9_v9-emergent-tuned-c282-01/
  step_121644_1788795070839078761-31814-0/evaluation.json
```

- 55 fresh, evaluation-owned artifacts, including 10 probe PNGs. Every length,
  FNV hash and SHA-256 matched actual bytes. Every mtime fell inside the invocation;
  a separate subsequent tool call rechecked all 55 hashes and fresh mtimes.
- The probe report matches the embedded report exactly. Decoded probe PNG channel
  means differ from float metrics by at most `0.000361526`, within 8-bit rounding
  tolerance. The ten probe PNGs are byte-identical to the old canonical images;
  equal deterministic bytes are freshly emitted at distinct, attributable paths.
- Autonomous offsets `32,64,96,128` all record `reference_fidelity = 0.0`,
  `micro_reference_drive_rms = 0.0`, and `macro_reference_drive_rms = 0.0`.
- Optimizer updates remained `30411 -> 30411`; development steps and optimizer
  windows completed were zero. Model, optimizer, world, manifest, training CSV,
  training metadata and all 35 old canonical artifacts retained their original
  bytes and mtimes (41 original files checked).
- Example new probe `001_5689be971a76e912_f0500_b3f000000_a0128.png` is 192x192,
  64,419 bytes, FNV `61a33ddc7ecfdae6`, with mtime
  `2026-09-07 15:33:08.297762050 UTC`. Its SHA-256 is
  `de595e2c8097b20517a286ec0ec94b7b5c404c2b2f3fc5bb571c84c8d8e3e1cd`.
- Tested binary SHA-256:
  `0e98f1d3854f661048c6470891f9cc7a9b2339886b99e9c7f68e5195f3d0cb12`.
  It was built before the fix commit, so its embedded build records base commit
  `c53d987e922a` and `dirty=true`; the tested production source is this fix.
- Validation: default build 81 unit tests plus 8 integration tests; OpenCL-feature
  build 88 unit tests plus 8 integration tests, followed by the newly added
  comparison regression (1 passed). Strict OpenCL-feature Clippy, formatting,
  diff checks and the locked release build passed.

Metadata/hash inspection and PNG decoding were sufficient; no Base64 image dumps
were used. The full run log, replay invocation and per-artifact verification JSON
were retained in `/data/data/com.termux/files/usr/tmp/titan-provenance-fix/`.
