# Router Variant Notes

## Current ALF Router

The ALF router preserves the Qwen3 MoE router forward contract:

1. Compute raw router logits.
2. Convert logits to router probabilities.
3. Add non-gradient expert bias only for top-k expert selection.
4. Gather selected expert weights from the original router probabilities.
5. Update expert bias in `torch.no_grad()` while training.

## Bias Update Policies

`proportional` updates each expert bias by the proportional load error:

```text
bias_delta = update_rate * (target_fraction - observed_fraction)
```

`sign` updates each expert bias with a fixed step in the load-error direction:

```text
bias_delta = update_rate * sign(target_fraction - observed_fraction)
```

`ema` applies the same proportional direction after smoothing the load error with
`bias_ema_beta`. `accumulated_sign` accumulates load error across
`update_interval` forwards and then applies one signed update.

`balanced_topk_sign` keeps the original signed ALF direction but updates only a
balanced subset of experts. It chooses the `bias_update_topk` largest positive
load errors and the same number of largest negative load errors, then applies:

```text
bias_delta[selected] = update_rate * sign(target_fraction - observed_fraction)
```

Use `proportional` as the default baseline. Use `sign` when the experiment should
test stronger discrete correction without scaling updates by the magnitude of load
imbalance. Use `balanced_topk_sign` when the experiment should keep equal numbers
of upward and downward bias writes.

The traditional auxiliary-loss baseline keeps the original Qwen3 router and uses a
forward hook only to record expert load. That makes its load metrics comparable with
ALF runs without changing the baseline routing behavior.

## Adding a New Policy

Add the policy name to `AlfConfig.bias_update_policy`, implement it in the router
bias update branch, expose it through metrics, and add a unit test showing the exact
bias delta for a deterministic load pattern.
