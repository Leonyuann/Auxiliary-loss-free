"""Serializable metrics helpers for auxiliary-loss-free Qwen3 MoE routers."""

from __future__ import annotations

import json
import re
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


def compute_maxvio(counts: Tensor) -> float:
    """Compute maximal violation for one MoE layer.

    Args:
        counts: Per-expert activation counts for one layer.

    Returns:
        MaxVio value. Returns zero when no expert assignments exist.
    """

    values = counts.detach().to(dtype=torch.float32, device="cpu")
    total = float(values.sum().item())
    if values.numel() == 0 or total == 0.0:
        return 0.0
    expected = total / float(values.numel())
    return float((values.max().item() - expected) / expected)


def mean_maxvio(layer_counts: dict[str, Tensor]) -> float:
    """Average MaxVio across MoE layers.

    Args:
        layer_counts: Mapping from layer/router names to per-expert counts.

    Returns:
        Mean MaxVio across layers.
    """

    if not layer_counts:
        return 0.0
    return float(sum(compute_maxvio(counts) for counts in layer_counts.values()) / len(layer_counts))


def collect_expert_load_counts(model: nn.Module) -> dict[str, Tensor]:
    """Collect current per-layer expert load counts from tracked routers.

    Args:
        model: Model containing tracked Qwen3 MoE routers.

    Returns:
        Mapping from router names to count tensors.
    """

    layer_counts: dict[str, Tensor] = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "last_expert_load"):
            layer_counts[module_name] = getattr(module, "last_expert_load").detach().cpu().clone()
    return layer_counts


def add_layer_counts(destination: dict[str, Tensor], source: dict[str, Tensor]) -> None:
    """Accumulate expert load counts by layer name.

    Args:
        destination: Mutable accumulated counts.
        source: Counts to add.
    """

    for layer_name, counts in source.items():
        if layer_name not in destination:
            destination[layer_name] = counts.detach().cpu().clone()
        else:
            destination[layer_name] = destination[layer_name] + counts.detach().cpu()


def collect_bias_update_steps(model: nn.Module) -> dict[str, int]:
    """Collect the current ALF bias update counters by router name.

    Args:
        model: Model that may contain auxiliary-loss-free routers.

    Returns:
        Mapping from router names to their current bias-update step counts.
    """

    update_steps: dict[str, int] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, Qwen3MoeAuxiliaryLossFreeTopKRouter):
            update_steps[module_name] = int(module.bias_update_steps.item())
    return update_steps


def collect_bias_update_deltas(
    model: nn.Module,
    previous_update_steps: dict[str, int],
) -> tuple[dict[str, Tensor], int]:
    """Collect ALF bias deltas that were newly applied since the last check.

    Args:
        model: Model that may contain auxiliary-loss-free routers.
        previous_update_steps: Mutable router-name to update-counter mapping.

    Returns:
        A tuple of newly applied per-router bias deltas and total update events.

    Notes:
        The router only stores the latest delta, so callers should invoke this
        after each optimizer-step bias update when exact per-update tracking is needed.
    """

    bias_deltas: dict[str, Tensor] = {}
    update_events = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, Qwen3MoeAuxiliaryLossFreeTopKRouter):
            continue
        current_steps = int(module.bias_update_steps.item())
        previous_steps = previous_update_steps.get(module_name, current_steps)
        if current_steps > previous_steps:
            bias_deltas[module_name] = module.last_bias_delta.detach().cpu().clone()
            update_events += current_steps - previous_steps
        previous_update_steps[module_name] = current_steps
    return bias_deltas, update_events


def add_bias_update_deltas(destination: dict[str, Tensor], source: dict[str, Tensor]) -> None:
    """Accumulate bias update deltas by router name.

    Args:
        destination: Mutable accumulated deltas.
        source: Newly collected deltas.
    """

    for layer_name, delta in source.items():
        if layer_name not in destination:
            destination[layer_name] = delta.detach().to(dtype=torch.float32, device="cpu").clone()
        else:
            destination[layer_name] = destination[layer_name] + delta.detach().to(dtype=torch.float32, device="cpu")


def activation_matrix_from_counts(layer_counts: dict[str, Tensor]) -> tuple[Tensor, list[str]]:
    """Convert per-layer counts into a layer-by-expert fraction matrix.

    Args:
        layer_counts: Mapping from layer/router names to per-expert counts.

    Returns:
        A tuple of activation fraction matrix and row layer names.
    """

    layer_names = sorted(layer_counts, key=_layer_sort_key)
    rows: list[Tensor] = []
    max_experts = max((int(layer_counts[name].numel()) for name in layer_names), default=0)
    for layer_name in layer_names:
        counts = layer_counts[layer_name].detach().to(dtype=torch.float32, device="cpu")
        total = counts.sum().clamp_min(1.0)
        fractions = counts / total
        if counts.numel() < max_experts:
            fractions = torch.nn.functional.pad(fractions, (0, max_experts - counts.numel()))
        rows.append(fractions)
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32), []
    return torch.stack(rows), layer_names


def bias_update_matrix_from_deltas(layer_deltas: dict[str, Tensor]) -> tuple[Tensor, list[str]]:
    """Convert per-layer bias deltas into a layer-by-expert matrix.

    Args:
        layer_deltas: Mapping from router names to per-expert bias deltas.

    Returns:
        A tuple of bias-update matrix and row layer names.
    """

    layer_names = sorted(layer_deltas, key=_layer_sort_key)
    rows: list[Tensor] = []
    max_experts = max((int(layer_deltas[name].numel()) for name in layer_names), default=0)
    for layer_name in layer_names:
        delta = layer_deltas[layer_name].detach().to(dtype=torch.float32, device="cpu")
        if delta.numel() < max_experts:
            delta = torch.nn.functional.pad(delta, (0, max_experts - delta.numel()))
        rows.append(delta)
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32), []
    return torch.stack(rows), layer_names


def activation_rows_from_counts(
    layer_counts: dict[str, Tensor],
    *,
    step: int | None,
    split: str,
) -> list[dict[str, Any]]:
    """Create table rows for expert activation counts and fractions.

    Args:
        layer_counts: Mapping from layer/router names to per-expert counts.
        step: Optional training step.
        split: Metric split name.

    Returns:
        JSON/W&B table rows.
    """

    rows: list[dict[str, Any]] = []
    for layer_index, layer_name in enumerate(sorted(layer_counts, key=_layer_sort_key)):
        counts = layer_counts[layer_name].detach().to(dtype=torch.float32, device="cpu")
        total = float(counts.sum().item())
        for expert_index, count in enumerate(counts.tolist()):
            rows.append(
                {
                    "step": step,
                    "split": split,
                    "layer_index": layer_index,
                    "layer": layer_name,
                    "expert": expert_index,
                    "count": int(count),
                    "fraction": 0.0 if total == 0.0 else float(count / total),
                }
            )
    return rows


def bias_update_rows_from_deltas(
    layer_deltas: dict[str, Tensor],
    *,
    step: int | None,
) -> list[dict[str, Any]]:
    """Create table rows for per-expert bias update deltas.

    Args:
        layer_deltas: Mapping from router names to per-expert bias deltas.
        step: Optional training step.

    Returns:
        JSON/W&B table rows.
    """

    rows: list[dict[str, Any]] = []
    for layer_index, layer_name in enumerate(sorted(layer_deltas, key=_layer_sort_key)):
        delta = layer_deltas[layer_name].detach().to(dtype=torch.float32, device="cpu")
        for expert_index, value in enumerate(delta.tolist()):
            rows.append(
                {
                    "step": step,
                    "layer_index": layer_index,
                    "layer": layer_name,
                    "expert": expert_index,
                    "bias_delta": float(value),
                }
            )
    return rows


def serialize_activation_matrix(matrix: Tensor, layer_names: list[str]) -> dict[str, Any]:
    """Convert an activation matrix into JSON-serializable content.

    Args:
        matrix: Layer-by-expert activation fraction matrix.
        layer_names: Matrix row labels.

    Returns:
        Serializable activation matrix dictionary.
    """

    return {
        "layers": layer_names,
        "values": matrix.detach().cpu().tolist(),
    }


def _layer_sort_key(layer_name: str) -> tuple[Any, ...]:
    """Sort router layer names by numeric model layer when present.

    Args:
        layer_name: Router module name.

    Returns:
        Stable sort key.
    """

    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        return (1, layer_name)
    return (0, int(match.group(1)), layer_name)


def loss_breakdown(outputs: Any, model: nn.Module) -> dict[str, float]:
    """Split model outputs into total, LM, and auxiliary loss values.

    Args:
        outputs: Hugging Face model output with ``loss`` and optional ``aux_loss``.
        model: Model used to resolve the router auxiliary-loss coefficient.

    Returns:
        Loss dictionary with stable float values.
    """

    total_loss = float(outputs.loss.detach().float().item())
    aux_loss_value = getattr(outputs, "aux_loss", None)
    aux_loss = 0.0 if aux_loss_value is None else float(aux_loss_value.detach().float().item())
    aux_coef = float(
        getattr(model, "router_aux_loss_coef", getattr(getattr(model, "config", object()), "router_aux_loss_coef", 0.0))
    )
    aux_loss_scaled = aux_coef * aux_loss
    return {
        "loss": total_loss,
        "lm_loss": total_loss - aux_loss_scaled,
        "aux_loss": aux_loss,
        "aux_loss_scaled": aux_loss_scaled,
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
        "bias_ema_beta": float(getattr(router, "expert_bias_ema_beta", 0.9)),
        "bias_update_topk": int(getattr(router, "expert_bias_update_topk", 1)),
        "bias_update_schedule": getattr(router, "expert_bias_update_schedule", "constant"),
        "bias_update_schedule_steps": getattr(router, "expert_bias_update_schedule_steps", None),
        "bias_update_end_rate": float(getattr(router, "expert_bias_update_end_rate", 0.0)),
        "last_bias_update_rate": float(getattr(router, "last_bias_update_rate", torch.tensor(0.0)).item()),
        "training_steps": int(router.training_steps.item()),
        "bias_update_steps": int(router.bias_update_steps.item()),
        "load": summarize_expert_load(counts=router.last_expert_load),
        "bias": summarize_expert_bias(router.expert_bias),
        "last_bias_delta": [float(value) for value in router.last_bias_delta.detach().cpu().tolist()],
        "load_error_ema": [float(value) for value in router.load_error_ema.detach().cpu().tolist()],
        "load_error_accumulator": [
            float(value) for value in router.load_error_accumulator.detach().cpu().tolist()
        ],
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
    "activation_matrix_from_counts",
    "activation_rows_from_counts",
    "add_bias_update_deltas",
    "add_layer_counts",
    "bias_update_matrix_from_deltas",
    "bias_update_rows_from_deltas",
    "collect_auxiliary_loss_free_router_metrics",
    "collect_bias_update_deltas",
    "collect_bias_update_steps",
    "collect_expert_load_counts",
    "collect_router_metrics",
    "compute_maxvio",
    "compute_expert_load_counts",
    "load_balance_metrics",
    "loss_breakdown",
    "mean_maxvio",
    "serialize_activation_matrix",
    "summarize_auxiliary_loss_free_router",
    "summarize_expert_bias",
    "summarize_expert_load",
    "summarize_tracked_router",
]
