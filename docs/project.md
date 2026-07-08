# ALF Project Overview

## Project Goal

This project explores auxiliary-loss-free optimization methods for Mixture-of-Experts
language models. The first milestone is to reproduce an auxiliary-loss-free baseline
on Qwen3 MoE, then use that baseline to evaluate routing and load-balancing variants.

The initial implementation focuses on a small Qwen3 MoE configuration for causal
language model pretraining experiments. This keeps the baseline practical to run on
limited hardware while preserving the core MoE routing behavior needed for research.

## Research Background

Qwen3 includes dense and MoE model variants. In MoE language models, the router selects
which experts process each token. Traditional MoE training commonly adds an auxiliary
load-balancing loss to avoid expert collapse, but that auxiliary loss can interfere
with the language modeling objective.

The auxiliary-loss-free method keeps expert utilization balanced without adding a
router auxiliary loss to the training objective. The main idea is to maintain an
expert-wise bias that affects expert selection:

- The router computes the normal routing probabilities.
- A non-gradient expert bias is added only for top-k expert selection.
- Expert output weights are still taken from the original routing probabilities.
- The expert bias is updated outside backpropagation according to observed expert
  load, increasing bias for underused experts and decreasing it for overloaded experts.

This design separates load-balancing control from the model loss.

## Baseline Scope

The first baseline will implement auxiliary-loss-free routing for Qwen3 MoE using
Hugging Face Transformers as the model backend. The project will not initially
reimplement the whole Qwen3 architecture.

In scope for the first baseline:

- Qwen3 MoE causal language model training.
- A tiny or small Qwen3 MoE configuration for fast local validation.
- Auxiliary-loss-free router replacement for Qwen3 MoE layers.
- Python experiment configuration loaded into dataclass config objects.
- A minimal but usable training framework.
- Router load metrics and checkpoints.
- Resume support that restores model, optimizer, and scheduler state.
- W&B experiment tracking for loss, learning rate, auxiliary loss, PPL, MaxVio, and
  expert activation heatmaps.
- Documentation for setup, training, and experiment comparison.

Out of scope for the first baseline:

- Training official large Qwen3 MoE checkpoints such as 30B-A3B.
- SFT, RLHF, or serving/inference optimization.
- Distributed expert parallelism beyond data-parallel DDP training.
- Full custom Qwen3 model implementation.

## Technical Direction

The implementation will use a minimal-intrusion approach:

1. Load or initialize a Hugging Face Qwen3 MoE causal language model.
2. Replace each Qwen3 MoE router with an auxiliary-loss-free router.
3. Disable the traditional router auxiliary loss when auxiliary-loss-free routing is
   enabled.
4. Train with the standard causal language modeling objective.
5. Log language modeling loss and expert load-balancing metrics.

The auxiliary-loss-free router should preserve the original Qwen3 MoE forward
interface so that model loading, checkpoints, and training code remain compatible
with the Hugging Face model implementation.

## Planned Project Structure

The expected repository structure is:

```text
experiments/
  qwen3_moe_tiny_alf.py
  qwen3_moe_tiny_aux_loss.py
docs/
  project.md
src/
  alf/
    config.py
    data.py
    modeling.py
    router.py
    train.py
    metrics.py
scripts/
tests/
PROJECT.md
README.md
pyproject.toml
```

All project dependencies and commands should be managed through `uv`.

## Experiment Configuration

Experiments should be configured through Python files that export typed dataclass
objects. This keeps experiments reproducible while allowing normal Python
composition for research variants.

Each experiment file should export a variable named `config`:

```python
from alf.config import AlfConfig, DataConfig, ExperimentConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    model=ModelConfig(...),
    data=DataConfig(...),
    training=TrainingConfig(...),
    alf=AlfConfig(enabled=True, ...),
)
```

The main dataclass groups are:

- `ModelConfig`: Qwen3 MoE model name, local checkpoint path, or tiny model dimensions.
- `DataConfig`: training text files, tokenizer, block size, and packing behavior.
- `TrainingConfig`: batch size, learning rate, training steps, checkpoint path, dtype,
  gradient accumulation, linear warmup steps, logging interval, dataloader workers,
  gradient checkpointing, and DDP options.
- `AlfConfig`: whether auxiliary-loss-free routing is enabled, bias initialization,
  bias update rate, bias update policy, bias update rate schedule, and whether to
  disable the original router auxiliary loss.
- `EvalConfig`: validation interval, validation batch size, and sample cap.
- `WandbConfig`: W&B online/offline/disabled mode, entity/project, group, tags, and
  checkpoint artifact logging.

The CLI should support dotted overrides for short runs and simple sweeps:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py --training.max_steps 20
```

No core training options should be hard-coded in scripts when they can be represented
in the Python experiment file.

## Scaling Experiments

The first scaled experiment family targets local C4 pretraining with an approximately
324M-parameter, 16-expert Qwen3 MoE configuration. The C4 source is expected at
`/vepfs-mlp2/ylq/data/c4/en` as gzipped JSONL shards with a `text` field.
`scripts/prepare_c4_bpe_tokens.py` encodes train and validation shards into int32
token files using `/vepfs-mlp2/ylq/tokenizers/owt_bpe_32k`, so training can reuse
the memory-mapped token-file dataset path. The default C4 preparation budget is
10B new train tokens per invocation; existing token files are appended using the
metadata document count as the resume offset. This matches the 100k-step two-GPU
training defaults at a 65,536-token global batch.

Two-GPU runs use `torchrun` DDP. Rank zero owns console progress, local JSONL/W&B
logging, validation, and checkpoint writes. During DDP training, ALF routers reduce
expert load counts across ranks before updating bias, and auxiliary-loss router
load hooks reduce their tracked loads before logging, so rank-zero metrics are
global for both experiment families.

The scaled configs keep gradient checkpointing disabled for all three C4 baselines
to make speed and memory comparisons use the same mode. ALF variants also require
that setting because checkpoint backward recomputation would double-count routed
tokens before the optimizer-step bias update. Each scaled run clips gradients with
`max_grad_norm=1.0` and keeps AdamW optimizer state in FP32 when model parameters
are BF16; ALF sign and EMA update bias once per optimizer step from accumulated
global-batch expert load.

## Expected Commands

The target user-facing commands are:

```bash
uv run alf-train experiments/qwen3_moe_tiny_alf.py
uv run alf-train experiments/qwen3_moe_tiny_aux_loss.py
uv run alf-inspect-router --checkpoint outputs/qwen3_moe_tiny_alf/latest
```

The first command trains the auxiliary-loss-free baseline. The second command trains
the traditional auxiliary-loss comparison baseline. The third command inspects expert
load and router bias statistics from a checkpoint.

Auxiliary-loss baseline checkpoints also track expert load metrics, but they do not
include ALF bias statistics because no ALF router is installed.

Use `--wandb.enabled false` for local smoke tests. Real experiment runs default to
W&B online mode and expect `WANDB_ENTITY` and `WANDB_PROJECT`.

## Evaluation Metrics

The first evaluation should compare:

- Language modeling loss.
- Auxiliary loss and scaled auxiliary-loss contribution.
- Learning rate and gradient norm.
- Tokens per second.
- Expert load variance.
- Maximum expert load divided by minimum expert load.
- Router bias distribution.
- Bias update policy behavior.
- MaxVio_batch, rolling-100 MaxVio_batch, and validation MaxVio_global.
- Expert activation heatmaps over layer and expert dimensions.
- Stability of training across short smoke-test runs.

These metrics are enough to verify whether the training framework and routing method
are working before scaling experiments.

## References

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts: https://arxiv.org/abs/2408.15664
- Hugging Face Transformers Qwen3 MoE implementation: https://github.com/huggingface/transformers


## Megatron Core 8xA100 Extension

The next scaling target is a single-node 8xA100 80GB Megatron Core path for a
approximately 1B-parameter Qwen-style MoE. The default topology is TP=1, PP=1,
CP=1, EP=4, and DP=2. The model uses 24 routed experts and top-3 routing, which
places 6 experts on each expert-parallel rank.

The project now has typed `MegatronConfig` fields, 1B ALF/ALF-EMA/aux-loss
experiment configs, a scripted 8-GPU launch, and a Megatron-compatible ALF router
that returns dense probability and routing-map tensors. ALF load counts are reduced
over the expert-data-parallel group so expert-parallel shards are not double-counted
when EP=4 and DP=2.

`alf-megatron-train` now validates the topology, initializes Megatron Core
model-parallel groups, builds the GPT/MoE model, runs a minimal forward/backward
training loop with Megatron Core DDP/optimizer, applies ALF bias updates after
optimizer steps, and writes per-rank checkpoint shards. The batch validator enforces
`micro_batch_size * gradient_accumulation_steps * data_parallel_size ==
global_batch_size`, and the sampler shards over the expert-data-parallel domain.
The 8xA100 acceptance smoke still needs to be run on the target host before
treating the path as production-ready.
