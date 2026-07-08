"""Megatron Core training entry point for ALF MoE experiments."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F

from alf.config import ExperimentConfig, asdict, load_experiment_config, parse_config_args

_INITIALIZED_DISTRIBUTED = False


def megatron_parallel_world_size(config: ExperimentConfig) -> int:
    """Compute the expected Megatron world size from configured parallel degrees.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Product of TP, PP, CP, EP, and DP degrees.
    """

    megatron = config.megatron
    return (
        megatron.tensor_model_parallel_size
        * megatron.pipeline_model_parallel_size
        * megatron.context_parallel_size
        * megatron.expert_model_parallel_size
        * megatron.data_parallel_size
    )


def estimate_moe_total_parameters(config: ExperimentConfig) -> int:
    """Estimate total parameters for the configured Qwen-style MoE shape.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Approximate total trainable parameter count.
    """

    model = config.model
    hidden = int(model.hidden_size)
    intermediate = int(model.intermediate_size)
    layers = int(model.num_hidden_layers)
    experts = int(model.num_experts)
    vocab = int(model.vocab_size)
    kv_heads = int(model.num_key_value_heads)
    heads = int(model.num_attention_heads)
    head_dim = hidden // heads
    attention = hidden * hidden + 2 * hidden * kv_heads * head_dim + hidden * hidden
    dense_mlp = 3 * hidden * intermediate
    router = experts * hidden
    moe = experts * dense_mlp + router
    norms = 2 * hidden
    embeddings = vocab * hidden * 2
    return int(embeddings + layers * (attention + moe + norms))


def validate_megatron_config(config: ExperimentConfig) -> None:
    """Validate Megatron-specific experiment semantics.

    Args:
        config: Loaded experiment configuration.

    Raises:
        ValueError: If the Megatron config is inconsistent with the 8xA100 plan.
    """

    megatron = config.megatron
    if not megatron.enabled:
        raise ValueError("megatron.enabled must be true for alf-megatron-train.")
    degrees = {
        "tensor_model_parallel_size": megatron.tensor_model_parallel_size,
        "pipeline_model_parallel_size": megatron.pipeline_model_parallel_size,
        "context_parallel_size": megatron.context_parallel_size,
        "expert_model_parallel_size": megatron.expert_model_parallel_size,
        "data_parallel_size": megatron.data_parallel_size,
    }
    for name, value in degrees.items():
        if int(value) <= 0:
            raise ValueError(f"megatron.{name} must be positive, got {value}.")
    expected_world_size = megatron_parallel_world_size(config)
    if expected_world_size != 8:
        raise ValueError(
            "Megatron 8xA100 defaults require TP*PP*CP*EP*DP == 8, "
            f"got {expected_world_size}."
        )
    if "WORLD_SIZE" in os.environ:
        runtime_world_size = int(os.environ["WORLD_SIZE"])
        if runtime_world_size != expected_world_size:
            raise ValueError(
                "WORLD_SIZE does not match Megatron parallel degrees: "
                f"WORLD_SIZE={runtime_world_size}, expected={expected_world_size}."
            )
    if config.model.num_experts % megatron.expert_model_parallel_size != 0:
        raise ValueError(
            "model.num_experts must be divisible by megatron.expert_model_parallel_size; "
            f"got {config.model.num_experts} and {megatron.expert_model_parallel_size}."
        )
    if config.model.num_experts_per_tok != 3:
        raise ValueError(f"Megatron 1B defaults require top3 routing, got {config.model.num_experts_per_tok}.")
    if config.alf.enabled and megatron.recompute_granularity is not None:
        raise ValueError("ALF Megatron training requires megatron.recompute_granularity=None to avoid load double counts.")
    if megatron.global_batch_size % (megatron.data_parallel_size * megatron.micro_batch_size) != 0:
        raise ValueError(
            "megatron.global_batch_size must be divisible by "
            "data_parallel_size * micro_batch_size."
        )


def megatron_transformer_config_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """Build Megatron Core ``TransformerConfig`` keyword arguments.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Keyword arguments matching Megatron Core's transformer configuration.
    """

    model = config.model
    megatron = config.megatron
    return {
        "num_layers": model.num_hidden_layers,
        "hidden_size": model.hidden_size,
        "num_attention_heads": model.num_attention_heads,
        "num_query_groups": model.num_key_value_heads,
        "ffn_hidden_size": model.intermediate_size,
        "moe_ffn_hidden_size": model.intermediate_size,
        "num_moe_experts": model.num_experts,
        "moe_router_topk": model.num_experts_per_tok,
        "moe_aux_loss_coeff": 0.0 if config.alf.enabled else model.router_aux_loss_coef,
        "moe_token_dispatcher_type": megatron.moe_token_dispatcher_type,
        "tensor_model_parallel_size": megatron.tensor_model_parallel_size,
        "pipeline_model_parallel_size": megatron.pipeline_model_parallel_size,
        "expert_model_parallel_size": megatron.expert_model_parallel_size,
        "context_parallel_size": megatron.context_parallel_size,
        "sequence_parallel": megatron.tensor_model_parallel_size > 1,
        "params_dtype": _torch_dtype(config.model.torch_dtype),
        "bf16": config.model.torch_dtype.lower() in {"bfloat16", "bf16"},
        "fp16": config.model.torch_dtype.lower() in {"float16", "fp16"},
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "transformer_impl": "local",
        "moe_grouped_gemm": False,
        "moe_router_enable_expert_bias": config.alf.enabled,
        "moe_router_bias_update_rate": config.alf.bias_update_rate,
    }


def build_megatron_gpt_model(config: ExperimentConfig) -> Any:
    """Build a Megatron Core GPT/MoE model for the configured experiment.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A Megatron Core ``GPTModel`` instance.
    """

    _require_megatron_core()
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    transformer_config = TransformerConfig(**megatron_transformer_config_kwargs(config))
    layer_spec = get_gpt_layer_local_spec(
        num_experts=config.model.num_experts,
        moe_grouped_gemm=False,
        normalization="RMSNorm",
    )
    return GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=config.model.vocab_size,
        max_sequence_length=config.data.block_size,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    )


def write_megatron_config_snapshot(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """Write a rank-zero config snapshot for Megatron experiments.

    Args:
        config: Loaded experiment configuration.
        output_dir: Experiment output directory.

    Returns:
        Path to the written JSON file.
    """

    path = Path(output_dir) / "alf_experiment_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return path


def train(config_path: str | Path, overrides: list[str] | None = None) -> Path:
    """Run a Megatron Core ALF experiment.

    Args:
        config_path: Python experiment config path.
        overrides: Optional dotted CLI overrides.

    Returns:
        Path to the experiment output directory once the full loop is wired.

    Raises:
        ImportError: If ``megatron-core`` is not installed.
        RuntimeError: Until the full Megatron training loop is implemented.
    """

    config = load_experiment_config(config_path, overrides)
    validate_megatron_config(config)
    _require_megatron_core()

    output_dir = Path(config.training.output_dir)
    _init_torch_distributed_if_needed(config)
    if _is_global_rank_zero():
        write_megatron_config_snapshot(config, output_dir)
        (output_dir / "megatron_transformer_config.json").write_text(
            json.dumps(megatron_transformer_config_kwargs(config), indent=2, default=str),
            encoding="utf-8",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    _cleanup_torch_distributed_if_needed()
    raise RuntimeError(
        "Megatron Core configuration and model construction are wired, but the "
        "multi-GPU Megatron optimizer/schedule/checkpoint training loop is not "
        "implemented yet. Do not treat this entrypoint as a completed training run."
    )


def main() -> None:
    """Run the command-line Megatron training entry point."""

    config_path, overrides = parse_config_args()
    train(config_path, overrides)


def _require_megatron_core() -> None:
    """Ensure Megatron Core is importable before launching training.

    Raises:
        ImportError: If Megatron Core cannot be imported.
    """

    try:
        importlib.import_module("megatron.core")
    except ImportError as error:
        raise ImportError(
            "Megatron Core is required for alf-megatron-train. Install project "
            "dependencies with `uv sync` after adding megatron-core, or run in an "
            "NGC/Megatron environment that provides `megatron.core`."
        ) from error


def _init_torch_distributed_if_needed(config: ExperimentConfig) -> None:
    """Initialize torch.distributed for Megatron launch bookkeeping.

    Args:
        config: Loaded experiment configuration.
    """

    global _INITIALIZED_DISTRIBUTED
    if megatron_parallel_world_size(config) <= 1 or dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    _INITIALIZED_DISTRIBUTED = True


def _cleanup_torch_distributed_if_needed() -> None:
    """Destroy the process group created by this lightweight entry point."""

    global _INITIALIZED_DISTRIBUTED
    if _INITIALIZED_DISTRIBUTED and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        _INITIALIZED_DISTRIBUTED = False


def _is_global_rank_zero() -> bool:
    """Return whether this process owns global side effects."""

    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a torch dtype name for Megatron config construction.

    Args:
        name: User-facing dtype string.

    Returns:
        Torch dtype.
    """

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(name, torch.float32)


if __name__ == "__main__":
    main()
