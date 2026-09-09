# Focused recovery and saturation experiments

Recovery can now select damage cases, and training can opt into a pre-tanh write-logit penalty. Existing configurations preserve their defaults and checkpoint signatures. No long training or empirical saturation-remediation experiment was run for this implementation.

## Longer, cheaper recovery tests

In an existing frozen evaluation JSON, set these fields inside `experiment.panel`:

```json
{
  "recovery_cases": ["macro_noise", "macro_patch"],
  "recovery_horizon": 512,
  "clock_robustness": false
}
```

Keep the other required panel fields from the existing configuration. Omitting `recovery_cases` retains all six historical damage cases. An empty list disables physical damage cases; `clock_robustness` independently adds its clock-sequence case. Unknown or duplicate cases are rejected, and recovery horizons are bounded to 0–2048 steps. Zero disables recovery entirely.

Each selected case still runs paired guided and autonomous trajectories. Reports include requested development steps and actual cumulative macro updates, both per trajectory sample and at the endpoint. Selection is output-only and does not change training signatures. Use a checkpoint copy for frozen evaluation, as in the prior follow-up.

## Optional saturation penalty

Add this field to `destination.experiment` in a **new fork request**, alongside the existing normalization mode and other experiment settings:

```json
"saturation_penalty": { "weight": 0.0001, "threshold": 2.5 }
```

These numbers illustrate the configuration; they are not empirically validated settings. Keep `optimizer_diagnostics: true` to record the weighted window loss and whether the penalty was applied.

For micro/macro pre-tanh logits z, define `E(z) = mean(max(abs(z) - threshold, 0)^2)`. The added loss is:

```text
weight / (2 * BPTT) * sum_over_steps(E(micro_logits) + E(macro_logits))
```

Both heads contribute at every full-core tracked step, including macro writes on steps without a macro field update. The penalty is added after visual/flow loss weighting. Decoder-only windows, inference, and frozen panels create no penalty graph. The frozen CPU gradient probe includes the same BPTT-averaged penalty, under that probe's existing fully guided reference policy.

Omitting the field, using `null`, or setting weight to zero preserves the old numerical path and signatures, including existing differentiable-RMSNorm forks. Active weight and threshold are signed and recorded in fork/config provenance; changing them requires another explicit fork. Tensor names/shapes, write bounds, and CSV columns remain unchanged. The OpenCL renderer VJP is merged with the direct CPU penalty gradients.

The previously prepared `long-control-request.json` and `long-rmsnorm-request.json` remain normalization-only continuations. To test saturation, create a separately named request and destination, then use `titan_image fork REQUEST.json` followed by `titan_image --config-json DEST/config.json`.

## Validation and next decision

Tests cover old signatures, invalid settings, exact selected-versus-full recovery trajectories, macro-update accounting, finite differences, gradient descent when tanh's derivative is zero, identical forward values with the option enabled, and no penalty graph at zero weight or inference. CPU integration checks full-core versus decoder-only logging. CPU/OpenCL gradient parity is checked with both zero and positive penalty weights.

Next, measure a short matched saturation/control fork before committing to a long continuation. Gate extension on finite gradients, reduced write saturation, retained guided reconstruction, and better macro recovery. A correct penalty gradient alone does not prove improved learned behavior.

Validation completed: 93 CPU library tests plus 8 regression tests passed. The OpenCL parity test passed with both zero and positive penalty weights (maximum absolute gradient error approximately 1e-9). Strict locked/offline OpenCL Clippy, formatting, and diff checks passed. Logs are in `analysis/followup_code_2026-09-08/`.
