"""Megatron-compatible auxiliary-loss-free MoE router helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from alf.router import Qwen3MoeAuxiliaryLossFreeTopKRouter


def reduce_expert_load_counts(expert_load: Tensor, reduce_group: Any | None = None) -> Tensor:
    """Reduce expert-load counts over a selected Megatron process group.

    Args:
        expert_load: Local per-expert load counts.
        reduce_group: Process group that should contribute counts. Pass the
            Megatron TP/CP/DP group and intentionally exclude the EP group so
            expert-parallel shards are not counted repeatedly.

    Returns:
        A reduced tensor when distributed is initialized, otherwise a clone of
        the local counts.
    """

    reduced = expert_load.detach().clone()
    if not dist.is_available() or not dist.is_initialized() or reduce_group is None:
        return reduced
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=reduce_group)
    return reduced


class MegatronAuxiliaryLossFreeTopKRouter(Qwen3MoeAuxiliaryLossFreeTopKRouter):
    """Return Megatron-style routing probabilities and maps with ALF bias.

    This adapter preserves the ALF selection rule used by the Hugging Face
    Qwen3 router while exposing the Megatron MoE router forward contract: a
    dense probability tensor and a boolean token-to-expert routing map.

    Attributes:
        reduce_group: Optional Megatron TP/CP/DP process group used to aggregate
            expert loads before optimizer-step bias updates.
        layer_number: Layer number assigned by Megatron MoE layers.
    """

    def __init__(self, *args: Any, reduce_group: Any | None = None, **kwargs: Any) -> None:
        """Initialize a Megatron-compatible ALF router.

        Args:
            *args: Positional arguments forwarded to the base ALF router.
            reduce_group: Optional process group for load-count reductions.
            **kwargs: Keyword arguments forwarded to the base ALF router.
        """

        super().__init__(*args, **kwargs)
        self.reduce_group = reduce_group
        self.layer_number: int | None = None

    def set_layer_number(self, layer_number: int) -> None:
        """Store the Megatron layer number for metrics and logging.

        Args:
            layer_number: One-indexed Megatron transformer layer number.
        """

        self.layer_number = int(layer_number)

    def set_reduce_group(self, reduce_group: Any | None) -> None:
        """Set the process group used for expert-load reductions.

        Args:
            reduce_group: Megatron TP/CP/DP process group or ``None`` for local
                counts only.
        """

        self.reduce_group = reduce_group

    def _record_expert_load(self, router_indices: Tensor) -> None:
        """Record selected-expert counts without reducing across EP ranks.

        Args:
            router_indices: Selected expert indices with shape ``(tokens, top_k)``.
        """

        with torch.no_grad():
            local_load = torch.bincount(router_indices.reshape(-1), minlength=self.num_experts)
            local_load = local_load.to(device=self.last_expert_load.device, dtype=torch.long)
            expert_load = (
                reduce_expert_load_counts(local_load, self.reduce_group)
                if self.training
                else local_load.detach().clone()
            )
            self._set_load_statistics(expert_load)
            if self.training:
                self.accumulated_expert_load.add_(
                    expert_load.to(device=self.accumulated_expert_load.device, dtype=self.accumulated_expert_load.dtype)
                )

    def forward(self, hidden_states: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Route hidden states and return Megatron-style MoE tensors.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            padding_mask: Optional boolean padding mask. ``True`` entries are
                excluded from load accumulation and routing output.

        Returns:
            Tuple of ``(probs, routing_map)`` where both tensors have shape
            ``(tokens, num_experts)``. ``probs`` is zero for unselected experts.
        """

        flat_hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(flat_hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        biased_router_probs = router_probs + self.expert_bias.to(
            device=router_probs.device,
            dtype=router_probs.dtype,
        )
        _, router_indices = torch.topk(biased_router_probs, self.top_k, dim=-1)
        router_scores = router_probs.gather(dim=-1, index=router_indices)
        if self.norm_topk_prob:
            eps = torch.finfo(router_scores.dtype).eps
            router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(eps)
        router_scores = router_scores.to(router_logits.dtype)

        valid_mask = None
        if padding_mask is not None:
            valid_mask = ~padding_mask.reshape(-1).to(device=router_indices.device, dtype=torch.bool)
            self._record_expert_load(router_indices[valid_mask])
        else:
            self._record_expert_load(router_indices)

        probs = torch.zeros(
            flat_hidden_states.shape[0],
            self.num_experts,
            device=router_scores.device,
            dtype=router_scores.dtype,
        )
        probs.scatter_(dim=-1, index=router_indices, src=router_scores)
        routing_map = torch.zeros(
            flat_hidden_states.shape[0],
            self.num_experts,
            device=router_indices.device,
            dtype=torch.bool,
        )
        routing_map.scatter_(dim=-1, index=router_indices, src=torch.ones_like(router_indices, dtype=torch.bool))
        if valid_mask is not None:
            probs = probs * valid_mask.unsqueeze(-1)
            routing_map = routing_map & valid_mask.unsqueeze(-1)
        return probs, routing_map


def build_megatron_alf_router(**kwargs: Any) -> MegatronAuxiliaryLossFreeTopKRouter:
    """Build a Megatron-compatible ALF router from explicit keyword arguments.

    Args:
        **kwargs: Constructor arguments for ``MegatronAuxiliaryLossFreeTopKRouter``.

    Returns:
        Configured Megatron-compatible ALF router.
    """

    return MegatronAuxiliaryLossFreeTopKRouter(**kwargs)


__all__ = [
    "MegatronAuxiliaryLossFreeTopKRouter",
    "build_megatron_alf_router",
    "reduce_expert_load_counts",
]
