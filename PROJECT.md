# ALF Project Plan

## Purpose

This document tracks the staged implementation plan for the ALF project. It is meant
to guide future planning and sprint-level recommendations.

## Sprint 0: Project Foundation

Status: complete.

Passes: true.

Goal: make the repository ready for reproducible Qwen3 MoE experiments.

Deliverables:

- Create the Python package layout under `src/alf/`.
- Add project dependencies through `uv` and update `pyproject.toml` and `uv.lock`.
- Add Python experiment config loading with typed dataclass objects.
- Add initial experiment files for auxiliary-loss-free and auxiliary-loss baselines.
- Add dotted CLI overrides for quick local runs and simple sweeps.
- Add basic README setup instructions.
- Add pytest structure and one config-loading test.

Acceptance criteria:

- `uv sync` completes.
- `uv run pytest` runs successfully.
- A Python experiment file can be loaded into typed dataclass objects.
- CLI overrides such as `--training.max_steps 2` update the loaded config safely.

## Sprint 1: Auxiliary-Loss-Free Router Baseline

Status: complete.

Passes: true.

Goal: implement the core auxiliary-loss-free router for Qwen3 MoE.

Deliverables:

- Implement a Qwen3 MoE router compatible with the Hugging Face router interface.
- Add non-gradient expert bias buffers.
- Use `router_probs + expert_bias` for top-k expert selection.
- Use original `router_probs` for selected expert weights.
- Update expert bias in `torch.no_grad()` based on observed expert load.
- Add model patching utilities to replace Qwen3 MoE routers.
- Disable traditional router auxiliary loss when ALF is enabled.

Acceptance criteria:

- Tiny Qwen3 MoE forward pass works after router replacement.
- Router output shapes match the Hugging Face router contract.
- Router bias has no gradient.
- Unit tests prove top-k selection can change because of bias while weights still use original router probabilities.

## Sprint 2: Minimal Training Framework

Status: complete.

Passes: true.

Goal: train a tiny Qwen3 MoE causal language model end to end.

Deliverables:

- Implement local text dataset loading and fixed-length token packing.
- Implement causal LM training loop with gradient accumulation.
- Add checkpoint save and resume.
- Log loss, learning rate, throughput, expert load, and bias statistics.
- Add CLI entry point `alf-train`.
- Add a small local smoke-test dataset or fixture.

Acceptance criteria:

- `uv run alf-train experiments/qwen3_moe_tiny_alf.py` runs for a few steps.
- A checkpoint is written and can be resumed.
- Training logs include both language modeling loss and router load metrics.

## Sprint 3: Baseline Comparison

Status: complete.

Passes: true.

Goal: compare auxiliary-loss-free routing against the traditional auxiliary-loss baseline.

Deliverables:

- Add an auxiliary-loss baseline config.
- Ensure identical model, data, optimizer, and training settings across baseline runs except router balancing method.
- Add CLI entry point `alf-inspect-router`.
- Add a comparison report template under `docs/`.
- Document expected metrics and interpretation.

Acceptance criteria:

- Tiny ALF and auxiliary-loss baseline runs both complete.
- Router inspection reports expert load variance, max/min load ratio, and bias distribution.
- A short comparison note can be produced from saved logs.

## Sprint 4: Research Iteration Hooks

Status: complete.

Passes: true.

Goal: make the codebase convenient for later auxiliary-loss-free variants.

Deliverables:

- Add configurable bias update policies.
- Add experiment config fields for update rate, update interval, bias clipping, and warmup behavior.
- Add structured metric export for later plotting.
- Add tests for each bias update policy.
- Extend documentation with guidance for adding new router variants.

Acceptance criteria:

- New router policies can be selected from Python experiment files without editing training code.
- Existing ALF and auxiliary-loss baseline configs still run.
- Metrics are saved in a format suitable for later analysis.

## Current Default Recommendation

## Sprint 5: Experiment Observability

Status: complete.

Passes: true.

Goal: record the metrics needed to observe Table 2, Figure 3, and expert activation
behavior in W&B.

Deliverables:

- Add W&B config and logging helpers.
- Add validation PPL and MaxVio_global evaluation.
- Add MaxVio_batch and rolling-100 MaxVio logging.
- Add expert activation heatmaps and tables.
- Log regular training metrics including learning rate, loss, LM loss, auxiliary loss,
  gradient norm, and throughput.

Acceptance criteria:

- `uv run pytest` passes.
- `uv run alf-train ... --wandb.enabled false` records JSONL observability metrics.
- W&B disabled mode does not import or call W&B.
- W&B enabled mode logs stable scalar, table, heatmap, and artifact keys.

## Sprint 6: C4 300M Scaling

Status: complete.

Passes: true.

Goal: support the first scaled research experiments on two A100 GPUs with C4 data.

Deliverables:

- Add DDP-aware training with rank-zero logging, validation, and checkpoint writes.
- Add dataloader worker, pin-memory, drop-last, and gradient checkpointing config fields.
- Synchronize ALF router expert load across DDP ranks before bias updates.
- Add C4 JSON.GZ to int32 token-file preparation.
- Add 300M-family C4 ALF, ALF-EMA, and auxiliary-loss experiment configs with a 16-expert MoE shape.
- Add a script for appending 10B-token C4 preparation increments and running the C4 300M baseline family.

Acceptance criteria:

- `uv run pytest` passes.
- Existing tiny and OWT configs remain compatible with single-process training.
- C4 token preparation can encode local gzipped JSONL shards and write metadata.
- `torchrun --standalone --nproc_per_node=2 -m alf.train ...` launches DDP training.

## Current Default Recommendation

Run the C4 300M baseline family with `bash scripts/run_c4_300m_baselines.sh`, then
compare ALF sign, ALF EMA, and auxiliary-loss metrics in W&B and local JSONL logs.


## Sprint 7: Megatron Core 8xA100 1B MoE Path

Status: in progress.

Passes: false.

Goal: add a Megatron Core path for single-node 8xA100 80GB training with TP=1,
PP=1, CP=1, EP=4, DP=2, and top-3 MoE routing.

Delivered so far:

- Added `MegatronConfig` and dotted override support through the existing config system.
- Added 1B-family ALF, ALF-EMA, and auxiliary-loss Megatron experiment configs.
- Added a scripted 8-GPU launch wrapper.
- Added a Megatron-compatible ALF top-k router and expert-data-parallel load reducer.
- Added Megatron config validation, transformer-config generation, and GPTModel construction helpers.
- Added a Megatron Core training loop with Megatron Core DDP/optimizer, expert-data-parallel sampling, optimizer-step ALF bias updates, metrics, and complete per-rank checkpoint resume.
- Added successful-update `max_steps` semantics, native auxiliary-loss/load observations, distributed validation, stacked ALF count reduction, and log-cadence hot-path metrics.

Remaining work:

- Run the 8xA100 10-step acceptance check on the target A100 host.
- Run the full 8xA100 throughput benchmark and extended local-vs-Transformer-Engine numerical parity suite.

## Sprint 8: Stability-Normalized Adaptive EMA

Status: complete.

Passes: true.

Goal: isolate adaptive EMA memory from adaptive controller gain in controlled
104M OWT and 300M C4 experiments.

Deliverables:

- Add a gain-coupled adaptive EMA policy that reuses the persistent/oscillation
  beta estimator and controls the stability-normalized feedback gain.
- Synchronize fixed-EMA, adaptive-beta/fixed-rate, and adaptive-beta/gain-coupled
  configs across the 104M and 300M experiment families.
- Expose controller beta, realized update rate, and normalized feedback gain in
  local JSONL and W&B router metrics.
- Enable multi-GPU `torchrun` execution in the OWT baseline launcher and include
  world size plus gradient accumulation in token preparation.

Acceptance criteria:

- The gain-coupled policy preserves persistent/oscillation beta state exactly.
- Both scale launchers expose the same three-way ablation with seed overrides.
- `uv run pytest` and launcher syntax checks pass.

## Sprint 9: Adaptive Per-Expert Momentum

Status: complete.

Passes: true.

Goal: extend per-expert second-moment bias adaptation with an optional smoothed
load-error direction in controlled 104M OWT and 300M C4 experiments.

Deliverables:

- Add an `adaptive_per_expert_momentum` policy with checkpointed FP32 first and
  second moments.
- Expose momentum configuration and state through typed configs, router metrics,
  JSONL, W&B, and checkpoint inspection.
- Add tuned OWT and C4 experiment configs plus opt-in launcher branches.
- Audit the original policy formula and record the intended scale-specific training
  and controller defaults.

Acceptance criteria:

- Exact two-step tests verify first-moment and second-moment update ordering.
- Configuration tests preserve the intended scale-specific controller and training
  defaults.
- Tiny training, checkpoint restore, full pytest, compilation, and launcher syntax
  checks pass.

## Sprint 10: Megatron Adaptive Per-Expert Controllers

Status: complete.

Passes: true.

Goal: extend the per-expert second-moment and momentum controllers to the
approximately 1B-parameter Megatron Core experiment family.

Deliverables:

- Implement both adaptive per-expert policies in the Megatron Core router using
  complete expert-DP-reduced optimizer-step loads.
- Preserve policy-specific FP32 first/second-moment and effective-rate buffers in
  Megatron checkpoints and expose them through shared router metrics.
- Add matched 1B C4 experiment configs and isolated opt-in launcher branches for
  both policies.
- Preserve and document the tuned 104M OWT and 300M C4 controller defaults already
  used by their experiment launchers.

Acceptance criteria:

- Exact Megatron Core formula tests cover both policies and BF16 checkpoint restore.
- The 1B configs preserve the sign baseline model, data, training, evaluation,
  and parallel topology outside controller metadata and output paths.
- Full pytest, compilation, launcher syntax, and diff checks pass.
