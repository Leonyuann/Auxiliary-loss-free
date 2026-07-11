"""Megatron Core training entry point for ALF MoE experiments."""

from __future__ import annotations

import importlib
import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from alf.config import ExperimentConfig, asdict, load_experiment_config, parse_config_args
from alf.data import build_packed_text_dataset, causal_lm_collate
from alf.metrics import (
    activation_matrix_from_counts,
    activation_rows_from_counts,
    append_jsonl,
    collect_router_metrics,
    mean_maxvio,
    serialize_activation_matrix,
)
from alf.observability import (
    AllToAllProfiler,
    CudaStepTimer,
    MovingAverage,
    gpu_memory_metrics,
    summarize_moe_observability,
)
from alf.megatron_router import (
    MegatronCoreAuxiliaryLossFreeTopKRouter,
    reset_megatron_alf_router_loads,
    update_megatron_alf_router_biases,
)
from alf.train import _build_optimizer, _build_scheduler, _clip_or_measure_gradient_norm
from alf.wandb_logging import ExperimentLogger

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
    implied_global_batch_size = megatron_effective_global_batch_size(config)
    if implied_global_batch_size != megatron.global_batch_size:
        raise ValueError(
            "training.gradient_accumulation_steps must make "
            "micro_batch_size * data_parallel_size * gradient_accumulation_steps "
            f"equal megatron.global_batch_size; got {implied_global_batch_size} and {megatron.global_batch_size}."
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
        "moe_router_enable_expert_bias": False,
        "moe_router_score_function": "softmax",
        "moe_router_bias_update_rate": 0.0,
        "recompute_granularity": megatron.recompute_granularity,
    }


def build_megatron_layer_spec(config: ExperimentConfig) -> Any:
    """Build a Megatron GPT layer spec with optional ALF router replacement.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Megatron Core transformer layer spec.
    """

    _require_megatron_core()
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    layer_spec = get_gpt_layer_local_spec(
        num_experts=config.model.num_experts,
        moe_grouped_gemm=False,
        normalization="RMSNorm",
    )
    if config.alf.enabled:
        moe_submodules = layer_spec.submodules.mlp.keywords["submodules"]
        moe_submodules.router = partial(
            MegatronCoreAuxiliaryLossFreeTopKRouter,
            alf_config=config.alf,
        )
    return layer_spec


def build_megatron_gpt_model(config: ExperimentConfig) -> Any:
    """Build a Megatron Core GPT/MoE model for the configured experiment.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A Megatron Core ``GPTModel`` instance.
    """

    _require_megatron_core()
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    transformer_config = TransformerConfig(**megatron_transformer_config_kwargs(config))
    layer_spec = build_megatron_layer_spec(config)
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

    if megatron_parallel_world_size(config) > 1 and ("RANK" not in os.environ or "WORLD_SIZE" not in os.environ):
        raise RuntimeError("Launch Megatron training with torchrun so RANK and WORLD_SIZE are set.")

    output_dir = Path(config.training.output_dir)
    device = _resolve_megatron_device()
    _init_torch_distributed_if_needed(config, device)
    _init_megatron_model_parallel(config)
    _seed_megatron_model_parallel_rng(config)
    try:
        if _is_global_rank_zero():
            write_megatron_config_snapshot(config, output_dir)
            (output_dir / "megatron_transformer_config.json").write_text(
                json.dumps(megatron_transformer_config_kwargs(config), indent=2, default=str),
                encoding="utf-8",
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        _run_megatron_training_loop(config, output_dir, device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        _cleanup_megatron_model_parallel()
        _cleanup_torch_distributed_if_needed()
    return output_dir


def _run_megatron_training_loop(
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device | None = None,
) -> None:
    """Run Megatron training with successful-update step semantics.

    Args:
        config: Loaded Megatron experiment configuration.
        output_dir: Directory for metrics and checkpoint shards.
        device: Optional device already selected for this rank.

    Raises:
        RuntimeError: If optimizer updates skip repeatedly or checkpointing fails.
    """

    device = device or _resolve_megatron_device()
    model = build_megatron_gpt_model(config).to(device)
    model.train()
    if dist.is_available() and dist.is_initialized():
        from megatron.core.distributed import (
            DistributedDataParallel as MegatronDistributedDataParallel,
            DistributedDataParallelConfig,
        )

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=config.megatron.distributed_optimizer,
        )
        model = MegatronDistributedDataParallel(
            config=model.config,
            ddp_config=ddp_config,
            module=model,
        )
    metric_model = _unwrap_megatron_model(model)
    _install_megatron_load_observers(metric_model)

    dataset = build_packed_text_dataset(
        tokenizer=None,
        paths=config.data.train_files,
        block_size=config.data.block_size,
        max_train_samples=config.data.max_train_samples,
    )
    sampler = _build_megatron_sampler(dataset, config)
    loader = DataLoader(
        dataset,
        batch_size=config.megatron.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=causal_lm_collate,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=config.training.drop_last,
    )
    data_iter = _cycle(loader)
    optimizer = _build_megatron_training_optimizer(model, config)
    scheduler = _build_megatron_training_scheduler(optimizer, config)
    successful_step = 0
    attempt = 0
    resume_checkpoint = _resolve_megatron_resume_checkpoint(config, output_dir)
    if resume_checkpoint is not None:
        successful_step, attempt = _load_megatron_rank_checkpoint(
            resume_checkpoint, metric_model, optimizer, scheduler, config, device
        )

    metrics_path = output_dir / "metrics.jsonl"
    logger = ExperimentLogger(
        config.wandb if _is_global_rank_zero() else None,
        experiment_name=config.name,
        config=asdict(config),
    )
    step_timer = CudaStepTimer(device)
    step_time_window = MovingAverage(window_size=100)
    throughput_window = MovingAverage(window_size=100)
    maxvio_window = MovingAverage(window_size=100)
    all_to_all_profiler = AllToAllProfiler.from_env()
    consecutive_skips = 0
    max_consecutive_skips = 100
    best_eval_ppl: float | None = None
    best_eval_maxvio: float | None = None

    try:
        while successful_step < config.training.max_steps:
            attempt += 1
            target_step = successful_step + 1
            if sampler is not None:
                sampler.set_epoch(attempt - 1)
            should_observe = (
                target_step % config.training.log_every == 0
                or target_step == config.training.max_steps
                or all_to_all_profiler.enabled_for_step(target_step)
            )
            if should_observe:
                step_timer.start()
            all_to_all_profiler.start(target_step)
            _reset_megatron_load_observers(metric_model)
            if config.alf.enabled:
                reset_megatron_alf_router_loads(metric_model)
            optimizer.zero_grad(set_to_none=True)
            if _is_megatron_ddp(model):
                model.zero_grad_buffer()
            loss_total = torch.zeros((), device=device, dtype=torch.float32)
            tokens = 0
            for accumulation_index in range(config.training.gradient_accumulation_steps):
                batch = next(data_iter)
                input_ids, labels, position_ids, loss_mask, padding_mask = _prepare_megatron_batch(batch, device)
                sync_context = (
                    model.no_sync()
                    if _is_megatron_ddp(model)
                    and accumulation_index < config.training.gradient_accumulation_steps - 1
                    else _nullcontext()
                )
                with sync_context:
                    losses = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=None,
                        labels=labels,
                        loss_mask=loss_mask,
                        padding_mask=padding_mask,
                    )
                    loss = losses.float().mean() / config.training.gradient_accumulation_steps
                    loss.backward()
                loss_total.add_(loss.detach())
                tokens += int(input_ids.numel())
            if _is_megatron_ddp(model):
                model.finish_grad_sync()
            optimizer_step_successful, grad_norm = _step_megatron_optimizer(
                optimizer, model, config.training.max_grad_norm
            )
            if not optimizer_step_successful:
                consecutive_skips += 1
                _clear_megatron_aux_metrics()
                if should_observe:
                    step_time_ms = _reduce_max_scalar(step_timer.stop_ms(), device)
                    all_to_all_profiler.stop(step_time_ms)
                if _is_global_rank_zero() and (consecutive_skips == 1 or consecutive_skips % 10 == 0):
                    print(
                        f"Megatron optimizer skipped attempt {attempt}; "
                        f"successful_step remains {successful_step} "
                        f"({consecutive_skips} consecutive skips).",
                        flush=True,
                    )
                if consecutive_skips >= max_consecutive_skips:
                    raise RuntimeError(
                        f"Optimizer skipped {consecutive_skips} consecutive attempts at "
                        f"successful step {successful_step}; aborting to avoid an infinite loop."
                    )
                continue

            consecutive_skips = 0
            bias_update_events = _post_megatron_optimizer_step(
                True, metric_model, scheduler, alf_enabled=config.alf.enabled
            )
            successful_step += 1
            should_log = (
                successful_step % config.training.log_every == 0
                or successful_step == config.training.max_steps
            )
            raw_aux_loss = _consume_megatron_aux_loss(config, successful_step) if should_log else 0.0
            if not should_log:
                _clear_megatron_aux_metrics()

            if should_observe:
                step_time_ms = _reduce_max_scalar(step_timer.stop_ms(), device)
            else:
                step_time_ms = 0.0
            profile_metrics = all_to_all_profiler.stop(step_time_ms)
            if should_log:
                step_layer_counts = _collect_megatron_load_observers(metric_model)
                reduced_lm_loss = _reduce_expert_dp_scalar(loss_total, device)
                aux_loss_scaled = raw_aux_loss * float(config.model.router_aux_loss_coef)
                total_loss = reduced_lm_loss + aux_loss_scaled
                elapsed = max(step_time_ms / 1000.0, 1e-9)
                global_tokens = _megatron_global_tokens(tokens, config)
                maxvio_batch = mean_maxvio(step_layer_counts)
                activation_matrix, activation_layers = activation_matrix_from_counts(step_layer_counts)
                activation_matrix_json = serialize_activation_matrix(activation_matrix, activation_layers)
                activation_rows = activation_rows_from_counts(
                    step_layer_counts, step=successful_step, split="train"
                )
                tokens_per_second = global_tokens / elapsed
                system_metrics = _reduce_system_metrics(gpu_memory_metrics(device), device)
                system_metrics.update(
                    {
                        "step_time_ms": step_time_ms,
                        "step_time_ms_rolling_100": step_time_window.update(step_time_ms),
                        "tokens_per_sec": tokens_per_second,
                        "tokens_per_sec_rolling_100": throughput_window.update(tokens_per_second),
                    }
                )
                record: dict[str, Any] = {
                    "step": successful_step,
                    "attempt": attempt,
                    "train": {
                        "loss": total_loss,
                        "lm_loss": reduced_lm_loss,
                        "aux_loss": raw_aux_loss,
                        "aux_loss_scaled": aux_loss_scaled,
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "grad_norm": grad_norm,
                        "tokens_per_second": tokens_per_second,
                        "maxvio_batch": maxvio_batch,
                        "maxvio_batch_rolling_100": maxvio_window.update(maxvio_batch),
                        "bias_update_events": bias_update_events,
                        "optimizer_step_successful": True,
                        "optimizer_skipped_attempts": attempt - successful_step,
                    },
                    "system": system_metrics,
                    "moe": summarize_moe_observability(step_layer_counts),
                    "router": collect_router_metrics(metric_model),
                    "expert_activation": {
                        "train": {"matrix": activation_matrix_json, "rows": activation_rows}
                    },
                }
                if profile_metrics:
                    record["profile"] = profile_metrics
                if _is_global_rank_zero():
                    append_jsonl(metrics_path, record)
                    logger.log(record, step=successful_step)
                    logger.log_expert_activation_heatmap(
                        "train/expert_activation", activation_matrix, step=successful_step
                    )
                    logger.log_expert_activation_table(
                        "train/expert_activation", activation_rows, step=successful_step
                    )

            if _should_megatron_evaluate(successful_step, config):
                eval_record = _evaluate_megatron_model(metric_model, config, device, successful_step)
                if _is_global_rank_zero():
                    eval_scalars = {
                        key: value
                        for key, value in eval_record.items()
                        if not key.endswith("_matrix")
                        and not key.endswith("_rows")
                        and not key.endswith("_layers")
                        and not key.endswith("_matrix_json")
                    }
                    append_jsonl(
                        metrics_path,
                        {
                            "step": successful_step,
                            **eval_scalars,
                            "expert_activation": {
                                "eval": {
                                    "matrix": eval_record["eval/expert_activation_matrix_json"],
                                    "rows": eval_record["eval/expert_activation_rows"],
                                }
                            },
                        },
                    )
                    logger.log(eval_scalars, step=successful_step)
                    logger.log_expert_activation_heatmap(
                        "eval/expert_activation",
                        eval_record["eval/expert_activation_matrix"],
                        step=successful_step,
                    )
                    logger.log_expert_activation_table(
                        "eval/expert_activation",
                        eval_record["eval/expert_activation_rows"],
                        step=successful_step,
                    )
                    eval_ppl = float(eval_record["eval/ppl"])
                    eval_maxvio = float(eval_record["eval/maxvio_global"])
                    best_eval_ppl = eval_ppl if best_eval_ppl is None else min(best_eval_ppl, eval_ppl)
                    best_eval_maxvio = (
                        eval_maxvio if best_eval_maxvio is None else min(best_eval_maxvio, eval_maxvio)
                    )
                    logger.update_summary(
                        {
                            "best/eval_ppl": best_eval_ppl,
                            "best/eval_maxvio_global": best_eval_maxvio,
                            "final/eval_ppl": eval_ppl,
                            "final/eval_maxvio_global": eval_maxvio,
                        }
                    )

            if (
                successful_step % config.training.save_every == 0
                or successful_step == config.training.max_steps
            ):
                _save_megatron_rank_checkpoint(
                    output_dir,
                    metric_model,
                    optimizer,
                    scheduler,
                    successful_step,
                    config,
                    attempt=attempt,
                )
    finally:
        logger.finish()


def _unwrap_megatron_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the wrapped Megatron module when using Megatron DDP.

    Args:
        model: Plain or Megatron DDP-wrapped model.

    Returns:
        The underlying model module.
    """

    return getattr(model, "module", model)


def _is_megatron_ddp(model: torch.nn.Module) -> bool:
    """Return whether a module is Megatron Core DDP-wrapped.

    Args:
        model: Module to inspect.

    Returns:
        Whether the module exposes Megatron DDP synchronization methods.
    """

    return all(hasattr(model, name) for name in ("zero_grad_buffer", "finish_grad_sync", "no_sync"))


def megatron_effective_global_batch_size(config: ExperimentConfig) -> int:
    """Return the optimizer-step sample count implied by Megatron data loading.

    Args:
        config: Loaded experiment configuration.

    Returns:
        Micro batch times gradient accumulation times configured DP replicas.
    """

    return (
        config.megatron.micro_batch_size
        * config.training.gradient_accumulation_steps
        * config.megatron.data_parallel_size
    )


def _build_megatron_training_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> Any:
    """Build an optimizer compatible with the Megatron training wrapper.

    Args:
        model: Plain model for single-process runs or Megatron DDP-wrapped model.
        config: Loaded experiment configuration.

    Returns:
        A Megatron Core optimizer for Megatron DDP, otherwise the existing torch optimizer.
    """

    if not _is_megatron_ddp(model):
        return _build_optimizer(
            model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_state_dtype=config.training.optimizer_state_dtype,
        )

    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=config.training.learning_rate,
        min_lr=0.0,
        weight_decay=config.training.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        bf16=config.model.torch_dtype.lower() in {"bfloat16", "bf16"},
        fp16=config.model.torch_dtype.lower() in {"float16", "fp16"},
        params_dtype=_torch_dtype(config.model.torch_dtype),
        use_distributed_optimizer=config.megatron.distributed_optimizer,
        clip_grad=_megatron_clip_grad(config.training.max_grad_norm),
    )
    return get_megatron_optimizer(optimizer_config, [model], use_gloo_process_groups=False)


def _megatron_clip_grad(max_grad_norm: float | None) -> float:
    """Normalize clipping config for Megatron optimizers.

    Args:
        max_grad_norm: Optional maximum gradient norm.

    Returns:
        Positive clipping threshold, or ``0.0`` when clipping is disabled.
    """

    if max_grad_norm is None or max_grad_norm <= 0:
        return 0.0
    return float(max_grad_norm)


def _build_megatron_training_scheduler(optimizer: Any, config: ExperimentConfig) -> Any:
    """Build the LR scheduler used by the Megatron training loop.

    Args:
        optimizer: Torch optimizer or Megatron optimizer.
        config: Loaded experiment configuration.

    Returns:
        Scheduler object exposing ``step`` and ``get_last_lr``.
    """

    if hasattr(optimizer, "get_loss_scale"):
        return MegatronLearningRateScheduler(
            optimizer,
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            scheduler_type=config.training.scheduler_type,
        )
    return _build_scheduler(
        optimizer,
        config.training.learning_rate,
        config.training.warmup_steps,
        max_steps=config.training.max_steps,
        scheduler_type=config.training.scheduler_type,
    )


class MegatronLearningRateScheduler:
    """Minimal LR scheduler for Megatron optimizers.

    Attributes:
        optimizer: Megatron optimizer with torch-style ``param_groups``.
        learning_rate: Peak learning rate.
        warmup_steps: Number of warmup steps.
        max_steps: Total scheduled steps.
        scheduler_type: Normalized scheduler type.
        last_step: Number of completed scheduler steps.
        last_lr: Most recently applied learning rate.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        learning_rate: float,
        warmup_steps: int,
        max_steps: int,
        scheduler_type: str,
    ) -> None:
        """Initialize and apply the step-zero Megatron learning rate.

        Args:
            optimizer: Megatron optimizer with ``param_groups``.
            learning_rate: Peak learning rate.
            warmup_steps: Number of warmup steps.
            max_steps: Total scheduled steps.
            scheduler_type: Scheduler type, ``constant`` or ``cosine``.

        Raises:
            ValueError: If the scheduler type is unsupported.
        """

        normalized_type = scheduler_type.lower().replace("-", "_")
        if normalized_type not in {"constant", "cosine", "cosine_annealing"}:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type!r}")
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.scheduler_type = normalized_type
        self.last_step = 0
        self.last_lr = self._lr_for_step(0)
        self._apply_lr(self.last_lr)

    def step(self) -> None:
        """Advance one scheduler step and apply the new learning rate."""

        self.last_step += 1
        self.last_lr = self._lr_for_step(self.last_step)
        self._apply_lr(self.last_lr)

    def get_last_lr(self) -> list[float]:
        """Return the latest learning rate in torch scheduler format.

        Returns:
            Single-element list containing the latest LR.
        """

        return [self.last_lr]

    def state_dict(self) -> dict[str, Any]:
        """Return serializable scheduler state.

        Returns:
            Scheduler state dictionary.
        """

        return {"last_step": self.last_step, "last_lr": self.last_lr}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state and reapply the stored learning rate.

        Args:
            state_dict: State produced by ``state_dict``.
        """

        self.last_step = int(state_dict.get("last_step", 0))
        self.last_lr = float(state_dict.get("last_lr", self._lr_for_step(self.last_step)))
        self._apply_lr(self.last_lr)

    def _lr_for_step(self, step: int) -> float:
        """Return the learning rate for a zero-based scheduler step.

        Args:
            step: Scheduler step index.

        Returns:
            Scheduled learning rate.
        """

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.learning_rate * float(step + 1) / float(self.warmup_steps)
        if self.scheduler_type == "constant":
            return self.learning_rate
        decay_steps = max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, float(step - self.warmup_steps + 1) / float(decay_steps)))
        return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self, learning_rate: float) -> None:
        """Set all optimizer parameter-group learning rates.

        Args:
            learning_rate: LR to write into every parameter group.
        """

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate


def _post_megatron_optimizer_step(
    optimizer_step_successful: bool,
    metric_model: torch.nn.Module,
    scheduler: Any,
    *,
    alf_enabled: bool,
) -> int:
    """Run post-step hooks only after a real optimizer update.

    Args:
        optimizer_step_successful: Whether the optimizer actually updated parameters.
        metric_model: Unwrapped model used for ALF router inspection.
        scheduler: Learning-rate scheduler to advance after successful updates.
        alf_enabled: Whether ALF router bias updates are enabled.

    Returns:
        Number of ALF bias update events applied.
    """

    if not optimizer_step_successful:
        return 0
    bias_update_events = update_megatron_alf_router_biases(metric_model) if alf_enabled else 0
    scheduler.step()
    return bias_update_events


def _step_megatron_optimizer(optimizer: Any, model: torch.nn.Module, max_grad_norm: float | None) -> tuple[bool, float]:
    """Apply one optimizer step and report whether parameters were updated.

    Args:
        optimizer: Torch or Megatron optimizer.
        model: Model whose gradients may need torch-side clipping.
        max_grad_norm: Maximum norm for torch optimizers.

    Returns:
        Tuple of optimizer-step success and measured or clipped gradient norm.
    """

    if not hasattr(optimizer, "get_loss_scale"):
        grad_norm = _clip_or_measure_gradient_norm(model, max_grad_norm)
        optimizer.step()
        return True, grad_norm

    step_result = optimizer.step()
    if isinstance(step_result, tuple):
        step_successful = bool(step_result[0]) if len(step_result) >= 1 else True
        grad_norm = float(step_result[1]) if len(step_result) >= 2 and step_result[1] is not None else 0.0
        return step_successful, grad_norm
    return True, 0.0


def _prepare_megatron_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare shifted GPT inputs for Megatron Core.

    Args:
        batch: Collated fixed-length token batch.
        device: Target device.

    Returns:
        Tuple of input ids, labels, position ids, loss mask, and padding mask.
    """

    tokens = batch["input_ids"].to(device, non_blocking=True)
    input_ids = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()
    batch_size, sequence_length = input_ids.shape
    position_ids = torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids, labels, position_ids, loss_mask, padding_mask


def _build_megatron_sampler(dataset: Any, config: ExperimentConfig) -> DistributedSampler | None:
    """Build a data-parallel sampler for Megatron training.

    Args:
        dataset: Training dataset.
        config: Loaded experiment configuration.

    Returns:
        Distributed sampler when distributed training is active, otherwise ``None``.
    """

    if not dist.is_available() or not dist.is_initialized():
        return None
    from megatron.core import parallel_state

    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_expert_data_parallel_world_size(),
        rank=parallel_state.get_expert_data_parallel_rank(),
        shuffle=True,
        seed=config.training.seed,
        drop_last=config.training.drop_last,
    )


def _install_megatron_load_observers(model: torch.nn.Module) -> None:
    """Attach detached load accumulation hooks to Megatron routers.

    Args:
        model: Unwrapped Megatron model containing top-k routers.
    """

    def observe_router_load(
        module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Accumulate the routing map produced by one router forward.

        Args:
            module: Router module that produced the output.
            _inputs: Positional forward inputs containing router hidden states.
            output: Dense routing probabilities and boolean routing map.
        """

        if not isinstance(output, tuple) or len(output) < 2 or not torch.is_tensor(output[1]):
            return
        routing_map = output[1]
        if routing_map.dtype is not torch.bool or routing_map.ndim != 2:
            return
        with torch.no_grad():
            module._alf_observed_expert_load.add_(
                routing_map.sum(dim=0).to(
                    device=module._alf_observed_expert_load.device,
                    dtype=torch.long,
                )
            )
            if module._alf_observe_probabilities:
                logits = module.gating(_inputs[0]).reshape(-1, module.config.num_moe_experts)
                probabilities = torch.softmax(logits.float(), dim=-1)
                module._alf_observed_probability_sum.add_(probabilities.sum(dim=0))

    for module in model.modules():
        if not module.__class__.__name__.endswith("TopKRouter"):
            continue
        num_experts = int(getattr(getattr(module, "config", None), "num_moe_experts", 0) or 0)
        if num_experts <= 0:
            continue
        if not hasattr(module, "_alf_observed_expert_load"):
            module.register_buffer(
                "_alf_observed_expert_load",
                torch.zeros(num_experts, dtype=torch.long, device=next(module.parameters()).device),
                persistent=False,
            )
            module.register_buffer(
                "_alf_observed_probability_sum",
                torch.zeros(num_experts, dtype=torch.float32, device=next(module.parameters()).device),
                persistent=False,
            )
            module._alf_observe_probabilities = False
        if not hasattr(module, "_alf_observer_handle"):
            module._alf_observer_handle = module.register_forward_hook(observe_router_load)


def _reset_megatron_load_observers(model: torch.nn.Module) -> None:
    """Reset detached router observations before a train or eval window.

    Args:
        model: Model containing installed router observation buffers.
    """

    for module in model.modules():
        counts = getattr(module, "_alf_observed_expert_load", None)
        if torch.is_tensor(counts):
            counts.zero_()
            module._alf_observed_probability_sum.zero_()


def _set_megatron_probability_observation(model: torch.nn.Module, enabled: bool) -> None:
    """Enable full router-probability accumulation for validation aux loss.

    Args:
        model: Model containing installed router observation buffers.
        enabled: Whether hooks should accumulate full softmax probabilities.
    """

    for module in model.modules():
        if hasattr(module, "_alf_observe_probabilities"):
            module._alf_observe_probabilities = bool(enabled)


def _collect_megatron_eval_aux_loss(model: torch.nn.Module, config: ExperimentConfig) -> float:
    """Compute raw validation auxiliary loss from global load/probability sums.

    Args:
        model: Model containing accumulated validation router observations.
        config: Experiment configuration defining experts, top-k, and aux mode.

    Returns:
        Mean raw load-balancing auxiliary loss across MoE layers.
    """

    if config.alf.enabled or float(config.model.router_aux_loss_coef) == 0.0:
        return 0.0
    observed = [
        (module._alf_observed_expert_load, module._alf_observed_probability_sum)
        for module in model.modules()
        if torch.is_tensor(getattr(module, "_alf_observed_expert_load", None))
    ]
    if not observed:
        return 0.0
    counts = torch.stack([item[0] for item in observed])
    probability_sums = torch.stack([item[1] for item in observed])
    if dist.is_available() and dist.is_initialized():
        from megatron.core import parallel_state

        group = parallel_state.get_expert_data_parallel_group()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(probability_sums, op=dist.ReduceOp.SUM, group=group)
    topk = float(config.model.num_experts_per_tok)
    experts = float(config.model.num_experts)
    tokens = counts.sum(dim=1).float() / topk
    raw_per_layer = (
        (counts.float() * probability_sums).sum(dim=1)
        * experts
        / (topk * tokens.square().clamp_min(1.0))
    )
    return float(raw_per_layer.mean().item())


def _collect_megatron_load_observers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Reduce all layer observations in one expert-DP collective.

    Args:
        model: Model containing accumulated expert-load observations.

    Returns:
        CPU expert-count tensors keyed by qualified router name.
    """

    observed = [
        (name, module._alf_observed_expert_load)
        for name, module in model.named_modules()
        if torch.is_tensor(getattr(module, "_alf_observed_expert_load", None))
    ]
    if not observed:
        return {}
    stacked = torch.stack([counts for _, counts in observed])
    if dist.is_available() and dist.is_initialized():
        from megatron.core import parallel_state

        dist.all_reduce(
            stacked,
            op=dist.ReduceOp.SUM,
            group=parallel_state.get_expert_data_parallel_group(),
        )
    return {name: stacked[index].detach().cpu() for index, (name, _) in enumerate(observed)}


def _clear_megatron_aux_metrics() -> None:
    """Clear Megatron's per-step MoE metrics without distributed reporting."""

    from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker

    get_moe_metrics_tracker().clear()


def _consume_megatron_aux_loss(config: ExperimentConfig, step: int) -> float:
    """Synchronize and return the raw Megatron router auxiliary loss.

    Args:
        config: Experiment configuration defining accumulation and aux mode.
        step: Successful optimizer step used by the metrics tracker.

    Returns:
        Mean raw load-balancing auxiliary loss across MoE layers.
    """

    if config.alf.enabled or float(config.model.router_aux_loss_coef) == 0.0:
        _clear_megatron_aux_metrics()
        return 0.0
    from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker

    totals: dict[str, float] = {}
    get_moe_metrics_tracker().report(
        loss_scale=1.0 / float(config.training.gradient_accumulation_steps),
        iteration=step,
        num_layers=config.model.num_hidden_layers,
        total_loss_dict=totals,
    )
    return float(totals.get("load_balancing_loss", 0.0))


def _reduce_expert_dp_scalar(value: torch.Tensor | float, device: torch.device) -> float:
    """Average a scalar over data replicas while excluding EP duplicates.

    Args:
        value: Local scalar value or scalar tensor.
        device: Device used for the reduction tensor.

    Returns:
        Expert-data-parallel mean as a Python float.
    """

    tensor = torch.as_tensor(value, device=device, dtype=torch.float32).detach().clone()
    if dist.is_available() and dist.is_initialized():
        from megatron.core import parallel_state

        group = parallel_state.get_expert_data_parallel_group()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor.div_(parallel_state.get_expert_data_parallel_world_size())
    return float(tensor.item())


def _should_megatron_evaluate(step: int, config: ExperimentConfig) -> bool:
    """Return whether validation is due after a successful optimizer step.

    Args:
        step: Completed successful optimizer step.
        config: Experiment configuration with validation cadence.

    Returns:
        Whether distributed Megatron validation should run.
    """

    if not config.data.validation_files or config.eval.eval_every <= 0:
        return False
    return step % config.eval.eval_every == 0 or step == config.training.max_steps


def _evaluate_megatron_model(
    model: torch.nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    """Evaluate Megatron LM loss, perplexity, MaxVio, and expert activation.

    Args:
        model: Unwrapped model participating in expert parallelism.
        config: Experiment configuration defining evaluation limits.
        device: Device used for validation tensors and collectives.
        step: Successful optimizer step associated with activation rows.

    Returns:
        Evaluation scalars and serializable expert-activation structures.
    """

    dataset = build_packed_text_dataset(
        tokenizer=None,
        paths=config.data.validation_files,
        block_size=config.data.block_size,
        max_train_samples=config.data.max_validation_samples,
    )
    if config.eval.max_eval_samples is not None:
        dataset = Subset(dataset, range(min(config.eval.max_eval_samples, len(dataset))))
    sampler = None
    if dist.is_available() and dist.is_initialized():
        from megatron.core import parallel_state

        sampler = DistributedSampler(
            dataset,
            num_replicas=parallel_state.get_expert_data_parallel_world_size(),
            rank=parallel_state.get_expert_data_parallel_rank(),
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.eval.eval_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=causal_lm_collate,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False,
    )
    was_training = model.training
    model.eval()
    _reset_megatron_load_observers(model)
    _set_megatron_probability_observation(model, True)
    total_loss_times_tokens = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            input_ids, labels, position_ids, loss_mask, padding_mask = _prepare_megatron_batch(batch, device)
            losses = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
                loss_mask=loss_mask,
                padding_mask=padding_mask,
            )
            tokens = int(loss_mask.sum().item())
            total_loss_times_tokens.add_(losses.float().mean().double() * tokens)
            total_tokens.add_(tokens)
    if dist.is_available() and dist.is_initialized():
        from megatron.core import parallel_state

        group = parallel_state.get_expert_data_parallel_group()
        dist.all_reduce(total_loss_times_tokens, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM, group=group)
    raw_aux_loss = _collect_megatron_eval_aux_loss(model, config)
    layer_counts = _collect_megatron_load_observers(model)
    _set_megatron_probability_observation(model, False)
    if was_training:
        model.train()
    eval_loss = float((total_loss_times_tokens / total_tokens.clamp_min(1)).item())
    aux_loss_scaled = raw_aux_loss * float(config.model.router_aux_loss_coef)
    matrix, layer_names = activation_matrix_from_counts(layer_counts)
    return {
        "eval/loss": eval_loss,
        "eval/ppl": math.exp(min(eval_loss, 20.0)),
        "eval/total_loss": eval_loss + aux_loss_scaled,
        "eval/aux_loss": raw_aux_loss,
        "eval/aux_loss_scaled": aux_loss_scaled,
        "eval/maxvio_global": mean_maxvio(layer_counts),
        "eval/tokens": int(total_tokens.item()),
        "eval/expert_activation_matrix": matrix,
        "eval/expert_activation_matrix_json": serialize_activation_matrix(matrix, layer_names),
        "eval/expert_activation_layers": layer_names,
        "eval/expert_activation_rows": activation_rows_from_counts(
            layer_counts, step=step, split="eval"
        ),
    }


def _checkpoint_topology(config: ExperimentConfig) -> dict[str, int]:
    """Return parallel degrees that determine checkpoint shard compatibility.

    Args:
        config: Experiment configuration containing parallel degrees.

    Returns:
        Mapping of topology dimension names to configured degrees.
    """

    megatron = config.megatron
    return {
        "tensor_model_parallel_size": int(megatron.tensor_model_parallel_size),
        "pipeline_model_parallel_size": int(megatron.pipeline_model_parallel_size),
        "context_parallel_size": int(megatron.context_parallel_size),
        "expert_model_parallel_size": int(megatron.expert_model_parallel_size),
        "data_parallel_size": int(megatron.data_parallel_size),
    }


def _resolve_megatron_resume_checkpoint(
    config: ExperimentConfig,
    output_dir: Path,
) -> Path | None:
    """Resolve the explicit or default Megatron checkpoint to resume.

    Args:
        config: Experiment configuration containing an optional explicit resume path.
        output_dir: Current experiment output directory.

    Returns:
        The explicit checkpoint path when configured, otherwise ``output_dir/latest``
        when it exists, or ``None`` when training should start from scratch.
    """

    if config.training.resume_from:
        return Path(config.training.resume_from)
    latest_checkpoint = output_dir / "latest"
    return latest_checkpoint if latest_checkpoint.exists() else None


def _distributed_optimizer_leaves(optimizer: Any) -> list[Any]:
    """Return optimizer leaves that own distributed parameter state.

    Args:
        optimizer: Megatron optimizer or recursively chained optimizer.

    Returns:
        Flat optimizer leaves in stable chain order.
    """

    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        return [optimizer]
    leaves: list[Any] = []
    for child in children:
        leaves.extend(_distributed_optimizer_leaves(child))
    return leaves


def _capture_distributed_optimizer_parameter_state(optimizer: Any) -> list[Any] | None:
    """Capture local FP32 master parameters and Adam moments when distributed.

    Args:
        optimizer: Torch or Megatron optimizer to checkpoint.

    Returns:
        Per-leaf reshardable states, or None for torch optimizers.
    """

    leaves = _distributed_optimizer_leaves(optimizer)
    if not any(hasattr(leaf, "get_parameter_state_dp_reshardable") for leaf in leaves):
        return None
    states = []
    for leaf in leaves:
        if not hasattr(leaf, "get_parameter_state_dp_reshardable"):
            states.append(None)
            continue
        state = leaf.get_parameter_state_dp_reshardable()
        for key, dtype_state in state.items():
            if not isinstance(key, int):
                continue
            for buckets in dtype_state.values():
                for bucket in buckets:
                    for parameter_state in bucket:
                        parameter_state.setdefault("padding", False)
        states.append(state)
    return states


def _restore_distributed_optimizer_parameter_state(
    optimizer: Any,
    parameter_states: list[Any] | None,
) -> None:
    """Restore distributed optimizer master parameters and moments.

    Args:
        optimizer: Megatron optimizer whose tensor state should be restored.
        parameter_states: Per-leaf states captured in the rank checkpoint.

    Raises:
        RuntimeError: If a distributed optimizer lacks compatible state.
    """

    leaves = _distributed_optimizer_leaves(optimizer)
    expects_parameter_state = any(
        hasattr(leaf, "load_parameter_state_from_dp_reshardable") for leaf in leaves
    )
    if not expects_parameter_state:
        return
    if parameter_states is None or len(parameter_states) != len(leaves):
        raise RuntimeError(
            "Checkpoint is missing compatible distributed optimizer parameter state."
        )
    for leaf, parameter_state in zip(leaves, parameter_states):
        if hasattr(leaf, "load_parameter_state_from_dp_reshardable"):
            with torch.no_grad():
                leaf.load_parameter_state_from_dp_reshardable(parameter_state)


def _save_megatron_rank_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    step: int,
    config: ExperimentConfig,
    *,
    attempt: int | None = None,
) -> None:
    """Atomically save one rank shard and publish complete metadata.

    Args:
        output_dir: Experiment output directory containing latest.
        model: Local model shard, including ALF or EMA router buffers.
        optimizer: Torch or Megatron optimizer for local parameters.
        scheduler: Learning-rate scheduler with serializable state.
        step: Number of completed successful optimizer updates.
        config: Experiment configuration used for topology metadata.
        attempt: Optional total optimizer attempt count.

    Raises:
        RuntimeError: If any expected rank shard is absent at publication.
    """

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = _world_size()
    checkpoint_dir = output_dir / "latest"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if _is_global_rank_zero():
        metadata = {
            "complete": False,
            "step": step,
            "attempt": int(attempt if attempt is not None else step),
            "world_size": world_size,
            "topology": _checkpoint_topology(config),
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    shard_path = checkpoint_dir / f"rank_{rank:05d}.pt"
    temporary_path = checkpoint_dir / f".rank_{rank:05d}.pt.tmp"
    state: dict[str, Any] = {
        "step": step,
        "attempt": int(attempt if attempt is not None else step),
        "world_size": world_size,
        "topology": _checkpoint_topology(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_parameter_state": _capture_distributed_optimizer_parameter_state(optimizer),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(state, temporary_path)
    os.replace(temporary_path, shard_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if _is_global_rank_zero():
        missing = [
            rank_index
            for rank_index in range(world_size)
            if not (checkpoint_dir / f"rank_{rank_index:05d}.pt").is_file()
        ]
        if missing:
            raise RuntimeError(f"Checkpoint is incomplete; missing rank shards: {missing}")
        (checkpoint_dir / "alf_experiment_config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
        metadata["complete"] = True
        metadata_path = checkpoint_dir / "metadata.json"
        temporary_metadata_path = checkpoint_dir / ".metadata.json.tmp"
        temporary_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_metadata_path, metadata_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _load_megatron_rank_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, int]:
    """Restore this rank from a complete topology-compatible checkpoint.

    Args:
        checkpoint_dir: Published Megatron checkpoint directory.
        model: Local model shard to restore.
        optimizer: Torch or Megatron optimizer to restore.
        scheduler: Learning-rate scheduler to restore.
        config: Current configuration for topology validation.
        device: Device used to map this rank's checkpoint tensors.

    Returns:
        Completed successful optimizer step and total attempt count.

    Raises:
        FileNotFoundError: If checkpoint metadata is absent.
        RuntimeError: If publication, shards, or model state are incomplete.
        ValueError: If world size or parallel topology differs.
    """

    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Megatron checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete", False):
        raise RuntimeError(f"Megatron checkpoint is incomplete: {checkpoint_dir}")
    expected_world_size = _world_size()
    if int(metadata.get("world_size", -1)) != expected_world_size:
        raise ValueError(
            f"Checkpoint world size {metadata.get('world_size')} does not match runtime "
            f"world size {expected_world_size}."
        )
    expected_topology = _checkpoint_topology(config)
    if metadata.get("topology") != expected_topology:
        raise ValueError(
            f"Checkpoint topology {metadata.get('topology')} does not match runtime "
            f"topology {expected_topology}."
        )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    shard_path = checkpoint_dir / f"rank_{rank:05d}.pt"
    missing = [
        rank_index
        for rank_index in range(expected_world_size)
        if not (checkpoint_dir / f"rank_{rank_index:05d}.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"Megatron checkpoint is incomplete; missing rank shards: {missing}")
    state = torch.load(shard_path, map_location=device, weights_only=False)
    if int(state.get("step", -1)) != int(metadata["step"]):
        raise RuntimeError(f"Rank shard step does not match checkpoint metadata: {shard_path}")
    if state.get("topology") != expected_topology:
        raise ValueError(f"Rank shard topology does not match runtime topology: {shard_path}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_model = [
        key for key in incompatible.missing_keys if not key.endswith("_alf_observed_expert_load")
    ]
    if unexpected or missing_model:
        raise RuntimeError(
            f"Incompatible model checkpoint; missing={missing_model}, unexpected={unexpected}"
        )
    optimizer.load_state_dict(state["optimizer"])
    _restore_distributed_optimizer_parameter_state(
        optimizer, state.get("optimizer_parameter_state")
    )
    scheduler.load_state_dict(state["scheduler"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device=device)
    return int(state["step"]), int(state.get("attempt", state["step"]))


def _cycle(loader: DataLoader) -> Any:
    """Yield batches from a dataloader forever.

    Args:
        loader: Source dataloader.

    Yields:
        Batches from repeated dataloader passes.
    """

    while True:
        for batch in loader:
            yield batch


def _reduce_scalar(value: float | int, device: torch.device, dtype: torch.dtype) -> float:
    """Reduce a scalar over all ranks.

    Args:
        value: Local scalar.
        device: Device for the temporary tensor.
        dtype: Tensor dtype.

    Returns:
        Reduced scalar value.
    """

    tensor = torch.tensor(value, device=device, dtype=dtype)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _reduce_max_scalar(value: float | int, device: torch.device) -> float:
    """Reduce a scalar by max over all Megatron ranks.

    Args:
        value: Local scalar value.
        device: Device for the temporary tensor.

    Returns:
        Maximum scalar value across ranks.
    """

    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _reduce_system_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """Reduce system metrics by max over all Megatron ranks.

    Args:
        metrics: Local system metric dictionary.
        device: Device for collective tensors.

    Returns:
        Metrics reduced with max across ranks.
    """

    return {key: _reduce_max_scalar(value, device) for key, value in metrics.items()}


def _megatron_global_tokens(local_tokens: int, config: ExperimentConfig) -> int:
    """Return optimizer-step tokens excluding expert-parallel duplicates.

    Args:
        local_tokens: Tokens processed by one rank over the local accumulated step.
        config: Loaded experiment configuration.

    Returns:
        Global tokens across configured data-parallel replicas.
    """

    return int(local_tokens) * int(config.megatron.data_parallel_size)


def _world_size() -> int:
    """Return distributed world size or one."""

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class _nullcontext:
    """Tiny context manager used to avoid importing contextlib in hot loops."""

    def __enter__(self) -> None:
        """Enter the context."""

        return None

    def __exit__(self, *args: Any) -> bool:
        """Exit the context.

        Args:
            *args: Exception details.

        Returns:
            ``False`` so exceptions propagate.
        """

        return False


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


def _init_torch_distributed_if_needed(
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> None:
    """Initialize torch.distributed for Megatron launch bookkeeping.

    Args:
        config: Loaded experiment configuration.
        device: CUDA device bound to the current local rank.
    """

    global _INITIALIZED_DISTRIBUTED
    if megatron_parallel_world_size(config) <= 1 or dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    kwargs = {"device_id": device} if backend == "nccl" and device is not None else {}
    dist.init_process_group(backend=backend, **kwargs)
    _INITIALIZED_DISTRIBUTED = True


def _init_megatron_model_parallel(config: ExperimentConfig) -> None:
    """Initialize Megatron Core model-parallel process groups.

    Args:
        config: Loaded experiment configuration.
    """

    from megatron.core import parallel_state

    if parallel_state.model_parallel_is_initialized():
        return
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=config.megatron.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.megatron.pipeline_model_parallel_size,
        context_parallel_size=config.megatron.context_parallel_size,
        expert_model_parallel_size=config.megatron.expert_model_parallel_size,
    )


def _seed_megatron_model_parallel_rng(config: ExperimentConfig) -> None:
    """Register Megatron CUDA RNG streams after model-parallel initialization.

    Args:
        config: Loaded experiment configuration containing the training seed.
    """

    if not torch.cuda.is_available():
        return
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(config.training.seed)


def _cleanup_megatron_model_parallel() -> None:
    """Destroy Megatron Core model-parallel process groups."""

    try:
        from megatron.core import parallel_state
    except ImportError:
        return
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()


def _resolve_megatron_device() -> torch.device:
    """Resolve and set the local Megatron device.

    Returns:
        Torch device for this process.
    """

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


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
