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
experiments/qwen3_moe_c4_1b_megatron_alf_adaptive_per_expert.py
experiments/qwen3_moe_c4_1b_megatron_alf_adaptive_per_expert_momentum.py
experiments/qwen3_moe_c4_1b_megatron_aux_loss.py
```

Scripted launch shape:

```bash
RUN_ALF=1 RUN_EMA=0 RUN_AUX=0 MAX_STEPS=10 bash scripts/run_c4_1b_megatron_8xa100.sh

RUN_ALF=0 RUN_EMA=0 RUN_AUX=0 RUN_ADAPTIVE_PER_EXPERT=1 \
  MAX_STEPS=10 bash scripts/run_c4_1b_megatron_8xa100.sh

RUN_ALF=0 RUN_EMA=0 RUN_AUX=0 RUN_ADAPTIVE_PER_EXPERT_MOMENTUM=1 \
  MAX_STEPS=10 bash scripts/run_c4_1b_megatron_8xa100.sh
```

The launch script also accepts LR-related overrides such as `LR`/`LEARNING_RATE`,
`WEIGHT_DECAY`, `WARMUP_STEPS`, `SCHEDULER_TYPE`, and `MAX_GRAD_NORM`, plus
`SAVE_EVERY` and `OUTPUT_ROOT` (`OUTPUT_DIR` remains a compatibility alias for the
output root). Every ALF controller and auxiliary-loss run writes to a distinct
experiment subdirectory below that root. Each branch automatically resumes its own complete
`latest` checkpoint when present, or starts from scratch when no checkpoint exists.
`MAX_STEPS` is the final successful optimizer-step target, not the number of
additional steps after resume.
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
scheduler, successful optimizer-step count, deterministic data cursor, Python/NumPy/
Torch RNG, and Megatron model-parallel CUDA RNG state. Checkpoints publish from a
staging directory and retain the prior complete checkpoint during rotation.
`training.max_steps` counts successful optimizer updates; overflow/skipped attempts
do not advance the scheduler, ALF bias, logging step, evaluation, or checkpoint step,
and 100 consecutive skips abort instead of looping silently.

Megatron training now reports native raw/scaled auxiliary loss, whole-global-batch
expert load for both auxiliary-loss and ALF runs, and distributed validation
loss/PPL/MaxVio/activation metrics. Router observation tensors and CUDA timing are
materialized only on logging steps, and all ALF layer counts are reduced in one
stacked expert-DP collective per optimizer update. Transformer Engine, grouped GEMM,
gradient-reduce overlap, and distributed-optimizer parameter-gather overlap are enabled by default. The local/sequential path remains
available through `MegatronConfig` overrides for debugging.

Two-A100 EP=2 Transformer Engine/grouped-GEMM smokes have completed ALF
training/evaluation/checkpoint save, a step-1-to-step-2 distributed-optimizer
resume, and an auxiliary-loss run with nonzero train/eval auxiliary loss and expert
loads. The full 8xA100 acceptance and throughput benchmark remain required before
long experiments.

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
- `adaptive_ema_variance`: lower EMA beta as batch-noise-corrected load
  variance increases.
- `adaptive_ema_persistent_oscillation`: lower beta for persistent load error
  and raise it for oscillating load error.
- `adaptive_ema_gain_coupled`: reuse the persistent/oscillation beta estimator
  and adapt the update rate to keep EMA-normalized feedback gain approximately
  constant, subject to safety clipping.
- `adaptive_per_expert`: keep an FP32 EMA of squared load error for each expert
  and scale its update by `base_rate / sqrt(second_moment + epsilon)`.
- `adaptive_per_expert_momentum`: add an FP32 EMA of load error and use it as
  the update direction with the same per-expert second-moment scaling.
- `accumulated_sign`: accumulate load error over an interval, then apply a sign step.
- `balanced_topk_sign`: update the most imbalanced positive and negative experts.

Bias update rate scheduling defaults to `--alf.bias_update_schedule constant`. Use
`linear` with `--alf.bias_update_schedule_steps` to decay from
`--alf.bias_update_rate` to `--alf.bias_update_end_rate` over post-warmup
optimizer steps. Set `--alf.bias_max_update_steps N` to allow updates through
optimizer step `N` and freeze bias from step `N + 1`; the default `None` keeps
updates enabled indefinitely.

The PyTorch baseline scripts expose the adaptive EMA policies as opt-in runs. For
the strict same-commit, same-seed three-way comparison (fixed beta/fixed rate,
adaptive beta/fixed rate, adaptive beta/gain-coupled rate), run:

```bash
RUN_ALF=0 RUN_AUX=0 RUN_EMA=1 \
RUN_ADAPTIVE_EMA_PERSISTENT_OSCILLATION=1 \
RUN_ADAPTIVE_EMA_GAIN_COUPLED=1 \
SEED=42 NPROC_PER_NODE=2 WANDB_GROUP=owt-104m-gain-coupled-ablation \
  bash scripts/run_owt_104m_baselines.sh

RUN_ALF=0 RUN_AUX=0 RUN_EMA=1 \
RUN_ADAPTIVE_EMA_PERSISTENT_OSCILLATION=1 \
RUN_ADAPTIVE_EMA_GAIN_COUPLED=1 \
SEED=42 NPROC_PER_NODE=2 WANDB_GROUP=c4-300m-gain-coupled-ablation \
  bash scripts/run_c4_300m_baselines.sh
```

Use `ALF_ADAPTIVE_BETA_MIN`, `ALF_ADAPTIVE_BETA_MAX`,
`ALF_ADAPTIVE_VARIANCE_REFERENCE`, and `ALF_ADAPTIVE_STATE_DECAY` to override
the shared adaptive defaults. Use `ALF_NORMALIZED_GAIN`, `ALF_GAIN_RATE_MIN`, and
`ALF_GAIN_RATE_MAX` for gain coupling. The comparison defaults are fixed EMA
`beta=0.5, rate=0.1`, adaptive beta in `[0.25, 0.75]` with fixed rate `0.1`, and
the same adaptive beta with target normalized gain `1/30` clipped to rates
`[0.05, 0.3]`. `BATCH_SIZE` is per process; the OWT launcher automatically
includes `NPROC_PER_NODE` and `GRADIENT_ACCUMULATION_STEPS` in its prepared token
budget. These three adaptive EMA policies remain specific to the Hugging Face/PyTorch
training path. The per-expert second-moment and momentum policies below support
both PyTorch and Megatron.

Run the per-expert controller as an opt-in baseline at either scale:

```bash
RUN_ALF=0 RUN_EMA=0 RUN_AUX=0 RUN_ADAPTIVE_PER_EXPERT=1 \
  bash scripts/run_owt_104m_baselines.sh

RUN_ALF=0 RUN_EMA=0 RUN_AUX=0 RUN_ADAPTIVE_PER_EXPERT=1 \
  bash scripts/run_c4_300m_baselines.sh
```

Set `RUN_ADAPTIVE_PER_EXPERT_MOMENTUM=1` instead to run the matched momentum
variant at either scale.

The launchers expose `ALF_ADAPTIVE_PER_EXPERT_BASE_RATE`,
`ALF_ADAPTIVE_PER_EXPERT_BETA`, `ALF_ADAPTIVE_PER_EXPERT_MOMENTUM_BETA`,
and `ALF_ADAPTIVE_PER_EXPERT_EPSILON`. The tuned PyTorch defaults use
base rate `1e-3` at both scales and `epsilon=1e-8`. OWT uses second-moment beta `0.6` without momentum; the
momentum run uses second-moment beta `0.9` and momentum beta `0.6`. C4 uses second-moment beta `0.9`, plus momentum beta `0.6` for the momentum
run. Router checkpoints and JSONL/W&B summaries preserve and expose FP32 per-expert first moments, second moments, and effective update rates.

Checkpoints include the experiment config in `alf_experiment_config.json`, so a copied
checkpoint directory can still be inspected with `alf-inspect-router`.

See [docs/project.md](docs/project.md) for the project design and [PROJECT.md](PROJECT.md)
for sprint-level development status.
