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

## C4 300M Experiments

Prepare local C4 JSON.GZ shards into reusable int32 token files and run the 300M-family
16-expert ALF, ALF-EMA, and auxiliary-loss baselines on two A100 GPUs:

```bash
bash scripts/run_c4_300m_baselines.sh
```

The script reads C4 from `/vepfs-mlp2/ylq/data/c4/en`, reuses
`/vepfs-mlp2/ylq/tokenizers/owt_bpe_32k`, and writes default token files under
`/vepfs-mlp2/ylq/data/c4/`. Re-running preparation appends another token budget
after the already processed C4 documents; use `C4_OVERWRITE=1` only when you want
to rebuild from the beginning. Set `RUN_ALF=0`, `RUN_EMA=0`, or `RUN_AUX=0` to skip
individual runs, and `RUN_PREPARE=0` to skip data preparation entirely. By default,
each preparation invocation targets 10B new train tokens, and the training configs
use 100k steps with a global batch of 65,536 tokens on two GPUs.

All three C4 baseline configs keep `training.gradient_checkpointing` disabled for
fair throughput comparisons; ALF and ALF-EMA also require it because checkpoint
recomputation would double-count routed tokens before the optimizer-step bias
update. The scaled configs use `max_grad_norm=1.0`,
FP32 AdamW optimizer state for BF16 parameters, and slower scheduled ALF bias
updates to make the 100k-step run less brittle.

Direct DDP launch example:

```bash
uv run torchrun --standalone --nproc_per_node=2 -m alf.train experiments/qwen3_moe_c4_300m_alf.py
```

## Megatron Core 1B MoE Plan

The repository now includes Megatron Core experiment configs for a single-node
8xA100 80GB target with TP=1, PP=1, CP=1, EP=4, and DP=2. The default 1B-family
shape uses 24 routed experts with top-3 activation, so each expert-parallel rank
owns 6 local experts.

Configs:

```bash
experiments/qwen3_moe_c4_1b_megatron_alf.py
experiments/qwen3_moe_c4_1b_megatron_alf_ema.py
experiments/qwen3_moe_c4_1b_megatron_aux_loss.py
```

Scripted launch shape:

```bash
RUN_ALF=1 RUN_EMA=0 RUN_AUX=0 MAX_STEPS=10 bash scripts/run_c4_1b_megatron_8xa100.sh
```

The Megatron entry point validates the 8-GPU topology, initializes Megatron
model-parallel groups, builds the Megatron GPT/MoE model, and runs a minimal
training loop with Megatron Core DDP/optimizer so expert and non-expert
gradients use the proper process groups. Data loading shards over the expert-data-parallel domain
(DP=2 for the default EP=4 topology), and ALF load counts reduce over that same
expert-DP domain before optimizer-step bias updates. The current environment has
not run the 8xA100 acceptance smoke, so treat that hardware validation as the next
required step before long experiments.

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
- `ema`: update bias from an exponential moving average of load error.
- `accumulated_sign`: accumulate load error over an interval, then apply a sign step.
- `balanced_topk_sign`: update the most imbalanced positive and negative experts.

Bias update rate scheduling defaults to `--alf.bias_update_schedule constant`. Use
`linear` with `--alf.bias_update_schedule_steps` to decay from
`--alf.bias_update_rate` to `--alf.bias_update_end_rate` over post-warmup
optimizer steps.

Checkpoints include the experiment config in `alf_experiment_config.json`, so a copied
checkpoint directory can still be inspected with `alf-inspect-router`.

See [docs/project.md](docs/project.md) for the project design and [PROJECT.md](PROJECT.md)
for sprint-level development status.
