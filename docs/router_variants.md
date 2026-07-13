# Router Variant Notes

## Current ALF Router

The ALF router preserves the Qwen3 MoE router forward contract:

1. Compute raw router logits.
2. Convert logits to router probabilities.
3. Add non-gradient expert bias only for top-k expert selection.
4. Gather selected expert weights from the original router probabilities.
5. Update expert bias in `torch.no_grad()` while training.

## Bias Update Policies

The paper does not clip expert bias values; standard experiment configs leave `bias_clip=None`.

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

`adaptive_ema_variance` computes a per-layer beta from the complete global
optimizer-step expert counts. For expert fractions `p`, uniform target `u`, `E`
experts, and `N` assignments:

```text
raw_variance = E * sum((u - p) ** 2)
batch_noise = (E - 1) / N
excess_variance = max(raw_variance - batch_noise, 0)
magnitude = excess_variance / (excess_variance + variance_reference)
beta = beta_max - (beta_max - beta_min) * magnitude
```

`adaptive_ema_persistent_oscillation` decomposes consecutive load errors into
persistent and oscillating components:

```text
persistent = (error_t + error_t_minus_1) / 2
oscillating = (error_t - error_t_minus_1) / 2
batch_noise = (E - 1) / N
persistent_signal = max(E * sum(persistent ** 2) - batch_noise / 2, 0)
oscillation_signal = max(E * sum(oscillating ** 2) - batch_noise / 2, 0)
persistent_energy = EMA(persistent_signal, energy_state_decay)
oscillation_energy = EMA(oscillation_signal, energy_state_decay)
beta = (oscillation_energy + batch_noise) /
       (persistent_energy + oscillation_energy + batch_noise)
```

The resulting beta is clipped to `bias_adaptive_beta_min` and
`bias_adaptive_beta_max`, then used in the standard load-error EMA. The default
variance reference is `2.5e-3`, energy-state decay is `0.9`, and beta bounds are
`0.1` and `0.95`. All adaptive state is stored as router buffers, preserved by
checkpoints, and kept in FP32 when model weights use BF16. Router metrics expose
the dynamic beta, raw/excess normalized variance, finite-batch noise, and the two
noise-corrected energy estimates for JSONL and W&B analysis.

`adaptive_ema_gain_coupled` is the first-stage adaptive controller. It deliberately
reuses the complete persistent/oscillation beta estimator above, but couples the
bias update rate to the selected beta:

```text
normalized_gain = update_rate * (1 - beta) / (1 + beta)
update_rate = clip(normalized_gain_target * (1 + beta) / (1 - beta),
                   rate_min, rate_max)
```

The synchronized 104M and 300M configs use `normalized_gain_target=1/30`,
`rate_min=0.05`, `rate_max=0.3`, and beta bounds `[0.25, 0.75]`. The target is
anchored to the fixed-EMA reference point because beta `0.5` and update rate
`0.1` produce normalized gain `1/30`. Rate clipping is a safety bound; when it
activates, the realized normalized gain is logged rather than reported as the
unclipped target. The policy requires a constant rate schedule because its rate
is already controlled online.

All adaptive policies are currently specific to the Hugging Face/PyTorch path.
Megatron configs continue to accept the existing fixed-beta `ema` policy only.

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

## Bias Update Rate Schedules

`bias_update_rate` is the initial bias learning rate `u`. The default
`bias_update_schedule="constant"` keeps `u` unchanged. Set
`bias_update_schedule="linear"` with `bias_update_schedule_steps` to linearly
decay `u` from `bias_update_rate` to `bias_update_end_rate` over post-warmup
optimizer steps; after the schedule length, the end rate is held.

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py \
  --alf.bias_update_schedule linear \
  --alf.bias_update_schedule_steps 1000 \
  --alf.bias_update_end_rate 0.0
```

Future schedules, such as cosine annealing, should be added through the same
router schedule branch so all bias update policies share the new rate behavior.

Set `bias_max_update_steps` to the last optimizer step that may change expert bias.
For example, `bias_max_update_steps=1000` freezes bias after step 1000. The default
`None` represents an infinite limit. This boundary uses the absolute optimizer-step
count, including warmup steps, and is restored from router checkpoint state.

The traditional auxiliary-loss baseline keeps the original Qwen3 router and uses a
forward hook only to record expert load. That makes its load metrics comparable with
ALF runs without changing the baseline routing behavior.

## Adding a New Policy

Add the policy name to `AlfConfig.bias_update_policy`, implement it in the router
bias update branch, expose it through metrics, and add a unit test showing the exact
bias delta for a deterministic load pattern.


## Megatron Router Adapter

`MegatronAuxiliaryLossFreeTopKRouter` adapts the same ALF selection rule to the
Megatron MoE router contract. It returns a dense probability tensor and boolean
routing map instead of the Hugging Face `(logits, scores, indices)` tuple. The
1B Megatron configs use top-3 routing with 24 experts.

For EP=4/DP=2 runs, expert-load reduction must use the Megatron TP/CP/DP group and
exclude the EP dimension. Reducing over the world group would double-count expert
parallel shards and corrupt both MaxVio metrics and ALF bias updates. Each router
accumulates local counts over all gradient-accumulation microbatches, then performs
one expert-DP reduction after a successful optimizer step before updating bias.
