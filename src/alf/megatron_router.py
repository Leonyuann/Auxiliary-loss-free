"""Megatron-compatible auxiliary-loss-free MoE router helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from alf.router import Qwen3MoeAuxiliaryLossFreeTopKRouter

try:
    from megatron.core.transformer.moe.router import TopKRouter
except ImportError:  # pragma: no cover - megatron-core is an optional runtime dependency.
    TopKRouter = None


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
    """Small standalone Megatron-style ALF router used by unit tests.

    This adapter preserves the ALF selection rule used by the Hugging Face
    Qwen3 router while exposing the Megatron MoE router forward contract: a
    dense probability tensor and a boolean token-to-expert routing map. The
    production Megatron Core model path uses
    ``MegatronCoreAuxiliaryLossFreeTopKRouter`` below.

    Attributes:
        reduce_group: Optional Megatron TP/CP/DP process group used to aggregate
            expert loads before optimizer-step bias updates.
        layer_number: Layer number assigned by Megatron MoE layers.
    """

    def __init__(self, *args: Any, reduce_group: Any | None = None, **kwargs: Any) -> None:
        """Initialize a standalone Megatron-compatible ALF router.

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
        """Accumulate local selected-expert counts without per-forward collectives.

        Args:
            router_indices: Selected expert indices with shape ``(tokens, top_k)``.
        """

        with torch.no_grad():
            if router_indices.dtype == torch.bool and router_indices.shape[-1] == self.num_experts:
                local_load = router_indices.sum(dim=0)
            else:
                local_load = torch.bincount(router_indices.reshape(-1), minlength=self.num_experts)
            local_load = local_load.to(device=self.last_expert_load.device, dtype=torch.long)
            self._set_load_statistics(local_load)
            if self.training:
                self.accumulated_expert_load.add_(
                    local_load.to(
                        device=self.accumulated_expert_load.device,
                        dtype=self.accumulated_expert_load.dtype,
                    )
                )

    def update_expert_bias_from_accumulated_load(self) -> bool:
        """Reduce optimizer-step counts once before applying the ALF update.

        Returns:
            Whether a bias update event occurred.
        """

        with torch.no_grad():
            reduced_load = reduce_expert_load_counts(self.accumulated_expert_load, self.reduce_group)
            self.accumulated_expert_load.copy_(
                reduced_load.to(
                    device=self.accumulated_expert_load.device,
                    dtype=self.accumulated_expert_load.dtype,
                )
            )
        return super().update_expert_bias_from_accumulated_load()

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
        return _softmax_alf_route(
            router_logits=router_logits,
            top_k=self.top_k,
            expert_bias=self.expert_bias,
            norm_topk_prob=self.norm_topk_prob,
            record_load=self._record_expert_load,
            padding_mask=padding_mask,
        )


if TopKRouter is not None:

    class MegatronCoreAuxiliaryLossFreeTopKRouter(TopKRouter):
        """Megatron Core MoE router with Qwen-style softmax ALF semantics.

        The router is constructed by Megatron ``MoELayer`` and therefore follows
        the native ``TopKRouter`` constructor. It intentionally does not use
        Megatron's built-in expert-bias mode because Megatron Core 0.18 restricts
        that mode to sigmoid/sqrtsoftplus scoring, while this project preserves
        Qwen/DDP softmax router probabilities.

        Attributes:
            expert_bias: Non-gradient ALF selection bias.
            accumulated_expert_load: Optimizer-step expert load accumulator.
            last_expert_load: Last observed per-expert load.
        """

        _FLOAT32_CONTROL_BUFFERS = (
            "expert_bias",
            "last_load_fraction",
            "last_bias_delta",
            "last_bias_update_rate",
            "load_error_ema",
            "load_error_accumulator",
        )

        def __init__(self, *args: Any, alf_config: Any | None = None, **kwargs: Any) -> None:
            """Initialize the Megatron Core ALF router.

            Args:
                *args: Positional arguments forwarded to Megatron ``TopKRouter``.
                alf_config: ALF dataclass or mapping used for update semantics.
                **kwargs: Keyword arguments forwarded to Megatron ``TopKRouter``.
            """

            pg_collection = kwargs.get("pg_collection")
            if pg_collection is None and len(args) >= 2:
                pg_collection = args[1]
            super().__init__(*args, **kwargs)
            self.alf_load_group = getattr(pg_collection, "expt_dp", None) or self.tp_dp_cp_group
            self._install_alf_state(alf_config)

        def _install_alf_state(self, alf_config: Any | None) -> None:
            """Install ALF buffers and hyperparameters on a Megatron router.

            Args:
                alf_config: ALF dataclass or mapping used for update semantics.
            """

            if hasattr(self, "expert_bias"):
                delattr(self, "expert_bias")
            self.norm_topk_prob = bool(_config_value(alf_config, "norm_topk_prob", True))
            self.expert_bias_update_rate = float(_config_value(alf_config, "bias_update_rate", 0.0))
            self.expert_bias_update_policy = str(_config_value(alf_config, "bias_update_policy", "proportional"))
            self.expert_bias_update_interval = int(_config_value(alf_config, "update_interval", 1))
            self.expert_bias_ema_beta = float(_config_value(alf_config, "bias_ema_beta", 0.9))
            self.expert_bias_update_topk = int(_config_value(alf_config, "bias_update_topk", 1))
            self.expert_bias_update_schedule = str(_config_value(alf_config, "bias_update_schedule", "constant"))
            self.expert_bias_update_schedule_steps = _config_value(alf_config, "bias_update_schedule_steps", None)
            self.expert_bias_update_end_rate = float(_config_value(alf_config, "bias_update_end_rate", 0.0))
            self.expert_bias_clip = _config_value(alf_config, "bias_clip", None)
            self.expert_bias_warmup_steps = int(_config_value(alf_config, "warmup_steps", 0))
            self.expert_bias_max_update_steps = _config_value(alf_config, "bias_max_update_steps", None)
            if self.expert_bias_update_schedule_steps is not None:
                self.expert_bias_update_schedule_steps = int(self.expert_bias_update_schedule_steps)
            if self.expert_bias_clip is not None:
                self.expert_bias_clip = float(self.expert_bias_clip)
            if self.expert_bias_max_update_steps is not None:
                self.expert_bias_max_update_steps = int(self.expert_bias_max_update_steps)
            self._validate_alf_state()

            num_experts = int(self.config.num_moe_experts)
            bias_init = float(_config_value(alf_config, "bias_init", 0.0))
            self.register_buffer("expert_bias", torch.full((num_experts,), bias_init, dtype=torch.float32))
            self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
            self.register_buffer("bias_update_steps", torch.zeros((), dtype=torch.long))
            self.register_buffer("last_expert_load", torch.zeros(num_experts, dtype=torch.long))
            self.register_buffer("accumulated_expert_load", torch.zeros(num_experts, dtype=torch.long))
            self.register_buffer("last_load_fraction", torch.zeros(num_experts, dtype=torch.float32))
            self.register_buffer("last_bias_delta", torch.zeros(num_experts, dtype=torch.float32))
            self.register_buffer("last_bias_update_rate", torch.zeros((), dtype=torch.float32))
            self.register_buffer("load_error_ema", torch.zeros(num_experts, dtype=torch.float32))
            self.register_buffer("load_error_accumulator", torch.zeros(num_experts, dtype=torch.float32))

        def _validate_alf_state(self) -> None:
            """Validate ALF hyperparameters installed on the Megatron router.

            Raises:
                ValueError: If any router hyperparameter is inconsistent.
            """

            if self.expert_bias_update_interval <= 0:
                raise ValueError("update_interval must be positive.")
            if self.expert_bias_update_topk <= 0:
                raise ValueError("bias_update_topk must be positive.")
            if self.expert_bias_update_topk > int(self.config.num_moe_experts):
                raise ValueError("bias_update_topk must not exceed num_moe_experts.")
            if self.expert_bias_warmup_steps < 0:
                raise ValueError("warmup_steps must be non-negative.")
            if self.expert_bias_max_update_steps is not None and self.expert_bias_max_update_steps < 0:
                raise ValueError("bias_max_update_steps must be non-negative or None.")
            if self.expert_bias_clip is not None and self.expert_bias_clip < 0.0:
                raise ValueError("bias_clip must be non-negative.")
            if not 0.0 <= self.expert_bias_ema_beta < 1.0:
                raise ValueError("bias_ema_beta must satisfy 0 <= beta < 1.")
            valid_policies = {"proportional", "sign", "ema", "accumulated_sign", "balanced_topk_sign"}
            if self.expert_bias_update_policy not in valid_policies:
                raise ValueError(f"Unsupported ALF policy: {self.expert_bias_update_policy!r}.")
            valid_schedules = {"constant", "linear"}
            if self.expert_bias_update_schedule not in valid_schedules:
                raise ValueError(f"Unsupported ALF schedule: {self.expert_bias_update_schedule!r}.")
            if self.expert_bias_update_schedule == "linear" and self.expert_bias_update_schedule_steps is None:
                raise ValueError("bias_update_schedule_steps is required for linear schedule.")

        def _apply(self, fn: Any, recurse: bool = True) -> "MegatronCoreAuxiliaryLossFreeTopKRouter":
            """Apply module conversions while keeping ALF control buffers fp32."""

            saved_buffers = {}
            for buffer_name in self._FLOAT32_CONTROL_BUFFERS:
                buffer = getattr(self, buffer_name, None)
                if torch.is_tensor(buffer) and buffer.is_floating_point():
                    saved_buffers[buffer_name] = buffer.detach().float().clone()
            super()._apply(fn, recurse=recurse)
            for buffer_name, saved_buffer in saved_buffers.items():
                converted_buffer = getattr(self, buffer_name, None)
                if torch.is_tensor(converted_buffer):
                    setattr(self, buffer_name, saved_buffer.to(device=converted_buffer.device, dtype=torch.float32))
            return self

        def _record_expert_load(self, routing_map: Tensor) -> None:
            """Record and accumulate local routing-map expert counts.

            Args:
                routing_map: Boolean token-to-expert map with shape ``(tokens, num_experts)``.
            """

            with torch.no_grad():
                local_load = routing_map.sum(dim=0).to(device=self.last_expert_load.device, dtype=torch.long)
                self._set_load_statistics(local_load)
                if self.training:
                    self.accumulated_expert_load.add_(
                        local_load.to(
                            device=self.accumulated_expert_load.device,
                            dtype=self.accumulated_expert_load.dtype,
                        )
                    )

        def _set_load_statistics(self, expert_load: Tensor) -> None:
            """Store per-expert load and fraction statistics.

            Args:
                expert_load: Per-expert assignment counts.
            """

            self.last_expert_load.copy_(expert_load.to(device=self.last_expert_load.device, dtype=torch.long))
            total_assignments = expert_load.sum()
            load_fraction = expert_load.to(dtype=torch.float32) / total_assignments.clamp_min(1).to(
                dtype=torch.float32
            )
            self.last_load_fraction.copy_(load_fraction.to(device=self.last_load_fraction.device))

        def reset_expert_load_accumulator(self) -> None:
            """Reset expert load accumulated for the current optimizer step."""

            with torch.no_grad():
                self.accumulated_expert_load.zero_()

        def update_expert_bias_from_accumulated_load(self) -> bool:
            """Reduce counts and update expert bias once for one optimizer step.

            Returns:
                Whether a bias update event occurred.
            """

            accumulated_load = reduce_expert_load_counts(
                self.accumulated_expert_load,
                self.alf_load_group,
            )
            return self.update_expert_bias_from_reduced_load(accumulated_load)

        def update_expert_bias_from_reduced_load(self, accumulated_load: Tensor) -> bool:
            """Update expert bias from counts already reduced over expert DP.

            Args:
                accumulated_load: Global-batch expert counts for this router.

            Returns:
                Whether a bias update event occurred.
            """

            with torch.no_grad():
                self.accumulated_expert_load.zero_()
                if int(accumulated_load.sum().item()) == 0:
                    self.last_bias_delta.zero_()
                    self.last_bias_update_rate.zero_()
                    return False
                self._set_load_statistics(accumulated_load.to(device=self.last_expert_load.device, dtype=torch.long))
                previous_updates = int(self.bias_update_steps.item())
                self._update_expert_bias()
                return int(self.bias_update_steps.item()) > previous_updates

        def _update_expert_bias(self) -> None:
            """Update the non-gradient expert bias from latest load statistics."""

            with torch.no_grad():
                self.training_steps.add_(1)
                self.last_bias_delta.zero_()
                self.last_bias_update_rate.zero_()
                if self.expert_bias_update_rate == 0.0:
                    return
                current_step = int(self.training_steps.item())
                if (
                    self.expert_bias_max_update_steps is not None
                    and current_step > self.expert_bias_max_update_steps
                ):
                    return
                if current_step <= self.expert_bias_warmup_steps:
                    return
                target_fraction = torch.full_like(self.last_load_fraction, 1.0 / float(self.config.num_moe_experts))
                load_error = target_fraction - self.last_load_fraction
                steps_after_warmup = current_step - self.expert_bias_warmup_steps
                update_rate = self._scheduled_bias_update_rate(steps_after_warmup)
                if self.expert_bias_update_policy == "accumulated_sign":
                    self.load_error_accumulator.add_(
                        load_error.to(device=self.load_error_accumulator.device, dtype=self.load_error_accumulator.dtype)
                    )
                    if steps_after_warmup % self.expert_bias_update_interval != 0:
                        return
                    bias_delta = update_rate * self.load_error_accumulator.sign()
                    self.load_error_accumulator.zero_()
                else:
                    if steps_after_warmup % self.expert_bias_update_interval != 0:
                        return
                    if self.expert_bias_update_policy == "sign":
                        bias_delta = update_rate * load_error.sign()
                    elif self.expert_bias_update_policy == "balanced_topk_sign":
                        bias_delta = update_rate * self._balanced_topk_sign(load_error)
                    elif self.expert_bias_update_policy == "ema":
                        self.load_error_ema.mul_(self.expert_bias_ema_beta).add_(
                            load_error.to(device=self.load_error_ema.device, dtype=self.load_error_ema.dtype),
                            alpha=1.0 - self.expert_bias_ema_beta,
                        )
                        bias_delta = update_rate * self.load_error_ema
                    else:
                        bias_delta = update_rate * load_error
                self.expert_bias.add_(bias_delta.to(device=self.expert_bias.device, dtype=self.expert_bias.dtype))
                if self.expert_bias_clip is not None:
                    self.expert_bias.clamp_(-self.expert_bias_clip, self.expert_bias_clip)
                self.last_bias_delta.copy_(
                    bias_delta.to(device=self.last_bias_delta.device, dtype=self.last_bias_delta.dtype)
                )
                self.last_bias_update_rate.fill_(float(update_rate))
                self.bias_update_steps.add_(1)

        def _scheduled_bias_update_rate(self, steps_after_warmup: int) -> float:
            """Compute the bias update rate for the current post-warmup step.

            Args:
                steps_after_warmup: One-indexed optimizer-step count after warmup.

            Returns:
                Scalar update rate.
            """

            if self.expert_bias_update_schedule == "constant":
                return self.expert_bias_update_rate
            schedule_steps = int(self.expert_bias_update_schedule_steps or 1)
            if schedule_steps <= 1:
                return self.expert_bias_update_end_rate
            progress = min(max((steps_after_warmup - 1) / float(schedule_steps - 1), 0.0), 1.0)
            return self.expert_bias_update_rate + progress * (
                self.expert_bias_update_end_rate - self.expert_bias_update_rate
            )

        def _balanced_topk_sign(self, load_error: Tensor) -> Tensor:
            """Select equal-size positive and negative top-k sign updates.

            Args:
                load_error: Per-expert target-minus-observed load error.

            Returns:
                Sparse sign vector for bias updates.
            """

            update_sign = torch.zeros_like(load_error)
            positive_indices = torch.nonzero(load_error > 0, as_tuple=False).flatten()
            negative_indices = torch.nonzero(load_error < 0, as_tuple=False).flatten()
            selected_per_side = min(
                self.expert_bias_update_topk,
                int(positive_indices.numel()),
                int(negative_indices.numel()),
            )
            if selected_per_side == 0:
                return update_sign
            positive_scores = load_error[positive_indices].abs()
            negative_scores = load_error[negative_indices].abs()
            positive_topk = positive_indices[torch.topk(positive_scores, k=selected_per_side).indices]
            negative_topk = negative_indices[torch.topk(negative_scores, k=selected_per_side).indices]
            update_sign[positive_topk] = 1.0
            update_sign[negative_topk] = -1.0
            return update_sign

        def routing(self, logits: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
            """Route logits with softmax ALF selection semantics.

            Args:
                logits: Router logits with shape ``(sequence, batch, num_experts)``.
                padding_mask: Optional boolean mask where ``True`` marks padding tokens.

            Returns:
                Tuple of dense routing probabilities and boolean routing map.
            """

            logits = logits.reshape(-1, self.config.num_moe_experts)
            padding = None if padding_mask is None else padding_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)
            probs, routing_map = _softmax_alf_route(
                router_logits=logits,
                top_k=self.topk,
                expert_bias=self.expert_bias,
                norm_topk_prob=self.norm_topk_prob,
                record_load=self._record_expert_load,
                padding_mask=padding,
            )
            return probs, routing_map

else:

    class MegatronCoreAuxiliaryLossFreeTopKRouter:  # pragma: no cover
        """Placeholder raised when Megatron Core is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Raise an import error for missing Megatron Core."""

            raise ImportError("megatron-core is required for MegatronCoreAuxiliaryLossFreeTopKRouter.")


def _softmax_alf_route(
    *,
    router_logits: Tensor,
    top_k: int,
    expert_bias: Tensor,
    norm_topk_prob: bool,
    record_load: Any,
    padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Apply Qwen-style ALF top-k routing to flattened logits.

    Args:
        router_logits: Flattened router logits with shape ``(tokens, num_experts)``.
        top_k: Number of selected experts per token.
        expert_bias: Non-gradient bias used only for expert selection.
        norm_topk_prob: Whether selected original probabilities are renormalized.
        record_load: Callback invoked with the valid routing map.
        padding_mask: Optional boolean mask where ``True`` marks padding tokens.

    Returns:
        Tuple of dense selected probabilities and boolean routing map.
    """

    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    biased_router_probs = router_probs + expert_bias.to(device=router_probs.device, dtype=router_probs.dtype)
    _, router_indices = torch.topk(biased_router_probs, top_k, dim=-1)
    router_scores = router_probs.gather(dim=-1, index=router_indices)
    if norm_topk_prob:
        eps = torch.finfo(router_scores.dtype).eps
        router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    router_scores = router_scores.to(router_logits.dtype)

    probs = torch.zeros_like(router_logits).scatter(1, router_indices, router_scores)
    routing_map = torch.zeros_like(router_logits, dtype=torch.bool).scatter(
        1,
        router_indices,
        torch.ones_like(router_indices, dtype=torch.bool),
    )
    if padding_mask is not None:
        valid_mask = ~padding_mask.to(device=routing_map.device, dtype=torch.bool)
        probs = probs * valid_mask.unsqueeze(-1)
        routing_map = routing_map & valid_mask.unsqueeze(-1)
    record_load(routing_map)
    return probs, routing_map


def _config_value(config: Any, name: str, default: Any) -> Any:
    """Read a value from a dataclass-like object or mapping.

    Args:
        config: Config object or mapping.
        name: Attribute/key name.
        default: Value returned when missing.

    Returns:
        Configured value or default.
    """

    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_megatron_alf_router(**kwargs: Any) -> MegatronAuxiliaryLossFreeTopKRouter:
    """Build a standalone Megatron-compatible ALF router.

    Args:
        **kwargs: Constructor arguments for ``MegatronAuxiliaryLossFreeTopKRouter``.

    Returns:
        Configured standalone router.
    """

    return MegatronAuxiliaryLossFreeTopKRouter(**kwargs)


def iter_megatron_alf_routers(module: torch.nn.Module) -> Any:
    """Yield Megatron Core ALF routers from a module tree.

    Args:
        module: Root module to traverse.

    Yields:
        Tuples of qualified module name and router module.
    """

    for name, child in module.named_modules():
        if isinstance(child, MegatronCoreAuxiliaryLossFreeTopKRouter):
            yield name, child


def reset_megatron_alf_router_loads(module: torch.nn.Module) -> None:
    """Reset all Megatron ALF router load accumulators.

    Args:
        module: Model containing Megatron ALF routers.
    """

    for _, router in iter_megatron_alf_routers(module):
        router.reset_expert_load_accumulator()


def update_megatron_alf_router_biases(module: torch.nn.Module) -> int:
    """Update all Megatron ALF router biases once per optimizer step.

    Args:
        module: Model containing Megatron ALF routers.

    Returns:
        Number of routers that applied a bias update.
    """

    routers = [router for _, router in iter_megatron_alf_routers(module)]
    grouped: dict[int, list[Any]] = {}
    for router in routers:
        grouped.setdefault(id(router.alf_load_group), []).append(router)

    update_events = 0
    for group_routers in grouped.values():
        stacked_load = torch.stack([router.accumulated_expert_load for router in group_routers])
        reduced_load = reduce_expert_load_counts(stacked_load, group_routers[0].alf_load_group)
        for index, router in enumerate(group_routers):
            if router.update_expert_bias_from_reduced_load(reduced_load[index]):
                update_events += 1
    return update_events


__all__ = [
    "MegatronAuxiliaryLossFreeTopKRouter",
    "MegatronCoreAuxiliaryLossFreeTopKRouter",
    "build_megatron_alf_router",
    "iter_megatron_alf_routers",
    "reduce_expert_load_counts",
    "reset_megatron_alf_router_loads",
    "update_megatron_alf_router_biases",
]
