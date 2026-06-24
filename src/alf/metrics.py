"""Serializable metrics helpers for auxiliary-loss-free Qwen3 MoE routers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .router import Qwen3MoeAuxiliaryLossFreeTopKRouter


def compute_expert_load_counts(selected_experts: Tensor, num_experts: int) -> Tensor:
    """Count how often each expert was selected.

    Args:
        selected_experts: Selected expert indices with shape `(..., top_k)`.
        num_experts: Number of experts to count.

    Returns:
        A tensor of shape `(num_experts,)` containing expert-selection counts.

    Raises:
        ValueError: If `num_experts` is not strictly positive.
    """

    if num_experts <= 0:
        msg = f"num_experts must be greater than zero, got {num_experts}."
        raise ValueError(msg)
    flattened = selected_experts.reshape(-1).to(dtype=torch.long)
    return torch.bincount(flattened, minlength=num_experts)


def summarize_expert_load(
    *,
    counts: Tensor | None = None,
    selected_experts: Tensor | None = None,
    num_experts: int | None = None,
) -> dict[str, Any]:
    """Summarize expert load into JSON-serializable statistics.

    Args:
        counts: Optional precomputed expert counts with shape `(num_experts,)`.
        selected_experts: Optional selected expert indices used to compute counts.
        num_experts: Required when `selected_experts` is provided instead of counts.

    Returns:
        A serializable dictionary with expert-count and load-balance statistics.

    Raises:
        ValueError: If neither counts nor selected experts are supplied.
    """

    if counts is None:
        if selected_experts is None or num_experts is None:
            msg = "Either counts or both selected_experts and num_experts must be provided."
            raise ValueError(msg)
        counts = compute_expert_load_counts(selected_experts, num_experts)

    counts = counts.detach().to(dtype=torch.float32, device="cpu")
    total_assignments = int(counts.sum().item())
    load_variance = float(counts.var(unbiased=False).item()) if counts.numel() else 0.0
    min_load = float(counts.min().item()) if counts.numel() else 0.0
    max_load = float(counts.max().item()) if counts.numel() else 0.0
    if min_load == 0.0:
        max_min_ratio = None if max_load > 0.0 else 0.0
    else:
        max_min_ratio = max_load / min_load

    return {
        "counts": [int(value) for value in counts.to(dtype=torch.long).tolist()],
        "total_assignments": total_assignments,
        "mean_load": float(counts.mean().item()) if counts.numel() else 0.0,
        "load_variance": load_variance,
        "min_load": int(min_load),
        "max_load": int(max_load),
        "max_min_load_ratio": None if max_min_ratio is None else float(max_min_ratio),
    }


def summarize_expert_bias(expert_bias: Tensor) -> dict[str, Any]:
    """Summarize expert-bias values into JSON-serializable statistics.

    Args:
        expert_bias: Expert bias tensor.

    Returns:
        A serializable dictionary with bias statistics.
    """

    bias = expert_bias.detach().to(dtype=torch.float32, device="cpu")
    return {
        "values": [float(value) for value in bias.tolist()],
        "mean": float(bias.mean().item()) if bias.numel() else 0.0,
        "std": float(bias.std(unbiased=False).item()) if bias.numel() else 0.0,
        "min": float(bias.min().item()) if bias.numel() else 0.0,
        "max": float(bias.max().item()) if bias.numel() else 0.0,
        "abs_max": float(bias.abs().max().item()) if bias.numel() else 0.0,
    }


def summarize_auxiliary_loss_free_router(
    router: Qwen3MoeAuxiliaryLossFreeTopKRouter,
) -> dict[str, Any]:
    """Summarize the latest load and bias state for a router instance.

    Args:
        router: Router instance to summarize.

    Returns:
        A serializable dictionary of router-specific metrics.
    """

    return {
        "num_experts": int(router.num_experts),
        "top_k": int(router.top_k),
        "bias_update_policy": router.expert_bias_update_policy,
        "training_steps": int(router.training_steps.item()),
        "bias_update_steps": int(router.bias_update_steps.item()),
        "load": summarize_expert_load(counts=router.last_expert_load),
        "bias": summarize_expert_bias(router.expert_bias),
        "last_bias_delta": [float(value) for value in router.last_bias_delta.detach().cpu().tolist()],
    }


def collect_auxiliary_loss_free_router_metrics(model: nn.Module) -> dict[str, Any]:
    """Collect load and bias summaries for every ALF router in a model tree.

    Args:
        model: Model containing auxiliary-loss-free routers.

    Returns:
        A serializable dictionary keyed by router name with aggregate summaries.
    """

    router_summaries: dict[str, Any] = {}
    aggregate_counts: list[Tensor] = []
    aggregate_bias: list[Tensor] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, Qwen3MoeAuxiliaryLossFreeTopKRouter):
            continue
        router_summaries[module_name] = summarize_auxiliary_loss_free_router(module)
        aggregate_counts.append(module.last_expert_load.detach().cpu())
        aggregate_bias.append(module.expert_bias.detach().cpu())

    metrics: dict[str, Any] = {
        "num_routers": len(router_summaries),
        "routers": router_summaries,
    }
    if aggregate_counts:
        metrics["aggregate_load"] = summarize_expert_load(counts=torch.stack(aggregate_counts).sum(dim=0))
    if aggregate_bias:
        metrics["aggregate_bias"] = summarize_expert_bias(torch.cat(aggregate_bias))
    return metrics


def summarize_tracked_router(module: nn.Module) -> dict[str, Any]:
    """Summarize a router that tracks expert load without ALF bias.

    Args:
        module: Router module with ``last_expert_load``.

    Returns:
        Serializable load summary.
    """

    return {
        "num_experts": int(getattr(module, "num_experts")),
        "top_k": int(getattr(module, "top_k")),
        "load": summarize_expert_load(counts=getattr(module, "last_expert_load")),
    }


def load_balance_metrics(load: Tensor) -> dict[str, float]:
    """Compute compatibility load-balancing metrics from expert counts.

    Args:
        load: One-dimensional expert load counts.

    Returns:
        Dictionary with stable names used by training logs and tests.
    """

    summary = summarize_expert_load(counts=load)
    return {
        "expert_load_mean": float(summary["mean_load"]),
        "expert_load_variance": float(summary["load_variance"]),
        "expert_load_min": float(summary["min_load"]),
        "expert_load_max": float(summary["max_load"]),
        "expert_load_max_min_ratio": float(summary["max_min_load_ratio"] or 0.0),
    }


def collect_router_metrics(model: nn.Module) -> dict[str, Any]:
    """Collect router metrics using the public training-log API.

    Args:
        model: Model that may contain auxiliary-loss-free routers.

    Returns:
        Serializable router metrics.
    """

    router_summaries: dict[str, Any] = {}
    aggregate_counts: list[Tensor] = []
    aggregate_bias: list[Tensor] = []

    for module_name, module in model.named_modules():
        if isinstance(module, Qwen3MoeAuxiliaryLossFreeTopKRouter):
            router_summaries[module_name] = summarize_auxiliary_loss_free_router(module)
            aggregate_counts.append(module.last_expert_load.detach().cpu())
            aggregate_bias.append(module.expert_bias.detach().cpu())
        elif hasattr(module, "last_expert_load") and hasattr(module, "num_experts") and hasattr(module, "top_k"):
            router_summaries[module_name] = summarize_tracked_router(module)
            aggregate_counts.append(module.last_expert_load.detach().cpu())

    metrics: dict[str, Any] = {
        "num_routers": len(router_summaries),
        "routers": router_summaries,
    }
    if aggregate_counts:
        metrics["aggregate_load"] = summarize_expert_load(counts=torch.stack(aggregate_counts).sum(dim=0))
    if aggregate_bias:
        metrics["aggregate_bias"] = summarize_expert_bias(torch.cat(aggregate_bias))
    return metrics


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file.

    Args:
        path: Destination JSONL path.
        record: JSON-serializable record.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = [
    "append_jsonl",
    "collect_auxiliary_loss_free_router_metrics",
    "collect_router_metrics",
    "compute_expert_load_counts",
    "load_balance_metrics",
    "summarize_auxiliary_loss_free_router",
    "summarize_expert_bias",
    "summarize_expert_load",
    "summarize_tracked_router",
]
