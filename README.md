# ALF

Auxiliary-loss-free routing experiments for Qwen3 MoE language models.

The project starts with a tiny Qwen3 MoE causal language modeling baseline so the
router behavior, training loop, checkpointing, and metrics can be validated locally
before scaling to larger experiments.

## Setup

Use `uv` for all project management:

```bash
uv sync
```

## Run Tests

```bash
uv run pytest
```

## Train Baselines

Auxiliary-loss-free baseline:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py
```

Traditional auxiliary-loss baseline:

```bash
uv run alf-train experiments/qwen3_moe_tiny_aux_loss.py
```

Short smoke run with CLI overrides:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py --training.max_steps 1 --wandb.enabled false
```

Inspect router load and bias metrics:

```bash
uv run alf-inspect-router --checkpoint outputs/qwen3_moe_tiny_alf/latest
```

## W&B Observability

Training runs log both local JSONL metrics and W&B metrics. By default, W&B is enabled
in online mode and reads `WANDB_ENTITY` and `WANDB_PROJECT` from the environment:

```bash
WANDB_ENTITY=my-team WANDB_PROJECT=alf uv run alf-train experiments/qwen3_moe_tiny_alf.py
```

Disable W&B for local smoke tests:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py --wandb.enabled false
```

Core W&B metric keys:

- `train/loss`, `train/lm_loss`, `train/aux_loss`, `train/aux_loss_scaled`
- `train/learning_rate`, `train/grad_norm`, `train/tokens_per_second`
- `train/maxvio_batch`, `train/maxvio_batch_rolling_100`
- `eval/loss`, `eval/ppl`, `eval/maxvio_global`
- `train/expert_activation/heatmap`, `eval/expert_activation/heatmap`

Resume from a checkpoint:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py --training.resume_from outputs/qwen3_moe_tiny_alf/latest
```

## Experiment Configs

Experiments are Python files under `experiments/` that export a typed
`ExperimentConfig` object. Dotted CLI overrides are supported for quick local runs:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py --alf.bias_update_policy sign --training.max_steps 2
```

Supported ALF bias update policies:

- `proportional`: update bias by the proportional load error.
- `sign`: update bias by fixed-size steps from the load error direction.

Checkpoints include the experiment config in `alf_experiment_config.json`, so a copied
checkpoint directory can still be inspected with `alf-inspect-router`.

See [docs/project.md](docs/project.md) for the project design and [PROJECT.md](PROJECT.md)
for sprint-level development status.
