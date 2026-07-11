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

The launch script also accepts LR-related overrides such as `LR`/`LEARNING_RATE`,
`WEIGHT_DECAY`, `WARMUP_STEPS`, `SCHEDULER_TYPE`, and `MAX_GRAD_NORM`.
The Docker image uses a CUDA 12.9 compiler/Python 3.12 base and builds Transformer
Engine 2.16.1 from the locked NVIDIA source. PyTorch and its CUDA 12.8 Python
dependencies remain locked by `uv.lock`. The launch script selects the `te-build`
dependency group explicitly so a runtime `uv` sync preserves Transformer Engine.

The Megatron entry point validates the configured topology, binds each CUDA device
before NCCL initialization, initializes model-parallel groups and RNG streams, then
builds the GPT/MoE model. Its Megatron Core DDP/optimizer path keeps expert and
non-expert gradients on the proper process groups. Data loading shards over the
expert-data-parallel domain (DP=2 for the default EP=4 topology), and ALF load
counts reduce over that same expert-DP domain before optimizer-step bias updates.
The Megatron path automatically resumes from `training.output_dir/latest` when that
checkpoint exists. `--training.resume_from CHECKPOINT` selects a different checkpoint
and takes precedence over the default. A checkpoint is published only after every
rank shard is present, and resume validates world size and TP/PP/CP/EP/DP topology
before restoring the model (including ALF/EMA router buffers), distributed optimizer,
scheduler, successful optimizer-step count, attempt count, and torch RNG state.
`training.max_steps` counts successful optimizer updates; overflow/skipped attempts
do not advance the scheduler, ALF bias, logging step, evaluation, or checkpoint step,
and 100 consecutive skips abort instead of looping silently.

Megatron training now reports native raw/scaled auxiliary loss, whole-global-batch
expert load for both auxiliary-loss and ALF runs, and distributed validation
loss/PPL/MaxVio/activation metrics. Router observation tensors and CUDA timing are
materialized only on logging steps, and all ALF layer counts are reduced in one
stacked expert-DP collective per optimizer update. Transformer Engine/grouped GEMM
and gradient-reduce overlap remain disabled until their numerical compatibility with
the custom softmax ALF router is accepted on the 8xA100 target.

Two-A100 EP=2 smokes have completed ALF training/evaluation/checkpoint save, a
step-1-to-step-2 distributed-optimizer resume, and an auxiliary-loss run with
nonzero train/eval auxiliary loss and expert loads. The full 8xA100 acceptance and
throughput benchmark remain required before long experiments.

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
- `system/step_time_ms`, `system/step_time_ms_rolling_100`, `system/tokens_per_sec`, `system/gpu_memory_allocated`
- `moe/expert_load_max_over_mean`, `moe/expert_load_cv`, `moe/expert_load_normalized_entropy`
- `moe/overflow_rate`, `moe/dropped_token_rate` are logged as `0.0` until a dispatcher exposes real overflow/drop counters
- `profile/all_to_all_time_ms`, `profile/all_to_all_time_ratio` when `ALF_PROFILE_ALL_TO_ALL_EVERY` and `ALF_PROFILE_ALL_TO_ALL_STEPS` enable profiling windows
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
optimizer steps. Set `--alf.bias_max_update_steps N` to allow updates through
optimizer step `N` and freeze bias from step `N + 1`; the default `None` keeps
updates enabled indefinitely.

Checkpoints include the experiment config in `alf_experiment_config.json`, so a copied
checkpoint directory can still be inspected with `alf-inspect-router`.

See [docs/project.md](docs/project.md) for the project design and [PROJECT.md](PROJECT.md)
for sprint-level development status.
