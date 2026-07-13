# Metrics and W&B Logging

## Paper Metrics

The auxiliary-loss-free paper reports two primary observability targets:

- Table 2: validation perplexity and MaxVio_global.
- Figure 3: MaxVio_batch over training steps, smoothed by 100 neighboring steps.

This project records both metrics during training.

## MaxVio

For one MoE layer, expected load is:

```text
expected_load = total_expert_assignments / num_experts
```

Layer MaxVio is:

```text
maxvio = (max_expert_load - expected_load) / expected_load
```

The model-level MaxVio is the average across MoE layers.

## W&B Keys

Common training metrics:

- `train/loss`: total backward loss.
- `train/lm_loss`: language-modeling loss with scaled auxiliary loss removed.
- `train/aux_loss`: raw router auxiliary loss when available.
- `train/aux_loss_scaled`: auxiliary loss multiplied by the model auxiliary-loss coefficient.
- `train/learning_rate`: scheduler learning rate.
- `train/grad_norm`: global gradient norm.
- `train/tokens_per_second`: training throughput.

Paper and routing metrics:

- `train/maxvio_batch`: current batch load-balance violation.
- `train/maxvio_batch_rolling_100`: rolling mean over the latest 100 JSONL/W&B logged batches.
- `eval/loss`: validation language-modeling loss.
- `eval/ppl`: validation perplexity.
- `eval/maxvio_global`: validation-set global load-balance violation.
- `train/expert_activation/heatmap`: layer-by-expert training activation heatmap.
- `eval/expert_activation/heatmap`: layer-by-expert validation activation heatmap.

Adaptive-controller router summaries are flattened below each router path. Important
fields include `adaptive_ema_beta`, `last_bias_update_rate`,
`normalized_feedback_gain`, `persistent_energy_ema`, `oscillation_energy_ema`,
`normalized_load_variance`, and `load_batch_noise`. The gain-coupled policy also
records its configured `gain_coupled_normalized_gain`, `gain_coupled_rate_min`, and
`gain_coupled_rate_max`. Compare `normalized_feedback_gain` with the target to see
when rate clipping is active.

## Local JSONL

Every run also writes `metrics.jsonl` under the experiment output directory. This
keeps scalar metrics, MaxVio values, expert activation matrices, and expert activation
table rows available even when W&B is disabled.

## Running

Online W&B:

```bash
WANDB_ENTITY=my-team WANDB_PROJECT=alf uv run alf-train experiments/qwen3_moe_tiny_alf.py
```

Disabled W&B:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py --wandb.enabled false
```


## Megatron metric cadence

Megatron uses the same public train/eval keys. Native router auxiliary loss is
reported as `train/aux_loss` and multiplied by the configured coefficient for
`train/aux_loss_scaled`; `train/loss` is LM loss plus that scaled term.
Validation reports the corresponding auxiliary fields and global expert loads.
Megatron auxiliary gradients use the optimizer loss scale divided by gradient
accumulation steps, matching the scaled LM backward call. Validation auxiliary loss
is computed per batch and token-weighted to match the DDP reporting definition.

To keep the hot path free of metric synchronization, detailed router, activation,
system-memory, and CUDA timing values are collected on `training.log_every` steps
and the final step. `attempt` and `train/optimizer_skipped_attempts` diagnose
mixed-precision overflow without changing the successful optimizer-step axis.
