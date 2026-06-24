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

All planned initial sprints are implemented. The next recommended planning step is
to define the first real research experiment: dataset choice, compute budget, model
scale, and which ALF policy variants to compare beyond the tiny smoke baseline.
