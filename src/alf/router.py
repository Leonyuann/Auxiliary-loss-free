"""Auxiliary-loss-free Qwen3 MoE router implementations."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_positive(name: str, value: int) -> None:
    """Validate that an integer hyperparameter is strictly positive.

    Args:
        name: Hyperparameter name for error reporting.
        value: Hyperparameter value to validate.

    Raises:
        ValueError: If the value is not strictly positive.
    """

    if value <= 0:
        msg = f"{name} must be greater than zero, got {value}."
        raise ValueError(msg)


class Qwen3MoeAuxiliaryLossFreeTopKRouter(nn.Module):
    """Apply auxiliary-loss-free top-k routing for Qwen3 MoE layers.

    The router matches the Hugging Face `Qwen3MoeTopKRouter` forward contract while
    adding a non-gradient expert bias that only affects expert selection.

    Attributes:
        top_k: Number of experts selected per token.
        num_experts: Total number of experts in the MoE layer.
        norm_topk_prob: Whether selected routing weights are renormalized.
        hidden_dim: Token hidden-state width.
        expert_bias_update_rate: Step size used when updating the expert bias.
        expert_bias_update_policy: Strategy used to convert load error into bias delta.
        expert_bias_adaptive_beta_min: Minimum adaptive EMA beta.
        expert_bias_adaptive_beta_max: Maximum adaptive EMA beta.
        expert_bias_adaptive_variance_reference: Excess normalized load variance
            at the midpoint of the variance-adaptive beta mapping.
        expert_bias_adaptive_state_decay: Decay for persistent and oscillation energies.
        expert_bias_update_interval: Number of optimizer steps between bias updates.
        expert_bias_update_topk: Number of positive-error and negative-error experts
            updated by the ``balanced_topk_sign`` policy.
        expert_bias_update_schedule: Schedule used for bias update rates.
        expert_bias_update_schedule_steps: Number of post-warmup optimizer steps
            used by the schedule.
        expert_bias_update_end_rate: Final bias update rate for scheduled decay.
        expert_bias_clip: Optional absolute clipping value for expert bias entries.
        expert_bias_warmup_steps: Number of optimizer steps to skip before updates.
        expert_bias_max_update_steps: Optional last optimizer step allowed to update
            expert bias. ``None`` allows updates indefinitely.
        weight: Router projection weights with shape `(num_experts, hidden_dim)`.
        expert_bias: Non-gradient bias applied only for top-k expert selection.
        training_steps: Number of optimizer-step bias update attempts seen by this router.
        bias_update_steps: Number of times the expert bias has been updated.
        last_expert_load: Last observed selected-expert counts.
        last_load_fraction: Last observed selected-expert fractions.
        last_bias_delta: Last applied expert-bias update delta.
    """

    _FLOAT32_CONTROL_BUFFERS = (
        "expert_bias",
        "last_load_fraction",
        "last_bias_delta",
        "last_bias_update_rate",
        "load_error_ema",
        "load_error_accumulator",
        "previous_load_error",
        "persistent_energy_ema",
        "oscillation_energy_ema",
        "last_adaptive_ema_beta",
        "last_normalized_load_variance",
        "last_excess_load_variance",
        "last_batch_noise",
    )

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        expert_bias_init: float = 0.0,
        expert_bias_update_rate: float = 0.0,
        expert_bias_update_policy: str = "proportional",
        expert_bias_update_interval: int = 1,
        expert_bias_ema_beta: float = 0.9,
        expert_bias_adaptive_beta_min: float = 0.1,
        expert_bias_adaptive_beta_max: float = 0.95,
        expert_bias_adaptive_variance_reference: float = 2.5e-3,
        expert_bias_adaptive_state_decay: float = 0.9,
        expert_bias_update_topk: int = 1,
        expert_bias_update_schedule: str = "constant",
        expert_bias_update_schedule_steps: int | None = None,
        expert_bias_update_end_rate: float = 0.0,
        expert_bias_clip: float | None = None,
        expert_bias_warmup_steps: int = 0,
        expert_bias_max_update_steps: int | None = None,
    ) -> None:
        """Initialize the auxiliary-loss-free router.

        Args:
            hidden_size: Hidden-state width consumed by the router.
            num_experts: Number of available experts.
            num_experts_per_tok: Number of experts selected for each token.
            norm_topk_prob: Whether selected routing weights are renormalized.
            expert_bias_init: Initial scalar value copied into all expert bias entries.
            expert_bias_update_rate: Update magnitude used for load-balancing bias steps.
            expert_bias_update_policy: Bias update policy. Supported values are
                `proportional`, `sign`, `ema`, `adaptive_ema_variance`,
                `adaptive_ema_persistent_oscillation`, `accumulated_sign`, and
                `balanced_topk_sign`.
            expert_bias_update_interval: Number of optimizer steps between updates.
            expert_bias_adaptive_beta_min: Minimum adaptive EMA beta.
            expert_bias_adaptive_beta_max: Maximum adaptive EMA beta.
            expert_bias_adaptive_variance_reference: Excess normalized load variance
                at the midpoint of the variance-adaptive beta mapping.
            expert_bias_adaptive_state_decay: Decay for persistent and oscillation
                energy estimates.
            expert_bias_update_topk: Number of positive-error and negative-error experts
                updated by the ``balanced_topk_sign`` policy.
            expert_bias_update_schedule: Schedule used for bias update rates.
                Supported values are ``constant`` and ``linear``.
            expert_bias_update_schedule_steps: Number of post-warmup optimizer steps
                used by the schedule. Required when using ``linear``.
            expert_bias_update_end_rate: Final bias update rate for scheduled decay.
            expert_bias_clip: Optional symmetric clip magnitude for bias entries.
            expert_bias_warmup_steps: Number of optimizer steps to skip before updates.
            expert_bias_max_update_steps: Optional last optimizer step allowed to
                update expert bias. ``None`` allows updates indefinitely.

        Raises:
            ValueError: If any hyperparameter is inconsistent.
        """

        super().__init__()
        _validate_positive("hidden_size", hidden_size)
        _validate_positive("num_experts", num_experts)
        _validate_positive("num_experts_per_tok", num_experts_per_tok)
        _validate_positive("expert_bias_update_interval", expert_bias_update_interval)
        _validate_positive("expert_bias_update_topk", expert_bias_update_topk)
        if expert_bias_update_topk > num_experts:
            msg = f"expert_bias_update_topk must be less than or equal to num_experts, got {expert_bias_update_topk}."
            raise ValueError(msg)
        if num_experts_per_tok > num_experts:
            msg = (
                "num_experts_per_tok must be less than or equal to num_experts, "
                f"got {num_experts_per_tok} and {num_experts}."
            )
            raise ValueError(msg)
        if expert_bias_warmup_steps < 0:
            msg = f"expert_bias_warmup_steps must be non-negative, got {expert_bias_warmup_steps}."
            raise ValueError(msg)
        if expert_bias_max_update_steps is not None and expert_bias_max_update_steps < 0:
            msg = (
                "expert_bias_max_update_steps must be non-negative or None, "
                f"got {expert_bias_max_update_steps}."
            )
            raise ValueError(msg)
        if expert_bias_clip is not None and expert_bias_clip < 0.0:
            msg = f"expert_bias_clip must be non-negative, got {expert_bias_clip}."
            raise ValueError(msg)
        if expert_bias_update_end_rate < 0.0:
            msg = f"expert_bias_update_end_rate must be non-negative, got {expert_bias_update_end_rate}."
            raise ValueError(msg)
        valid_policies = {
            "proportional",
            "sign",
            "ema",
            "adaptive_ema_variance",
            "adaptive_ema_persistent_oscillation",
            "accumulated_sign",
            "balanced_topk_sign",
        }
        if expert_bias_update_policy not in valid_policies:
            msg = (
                "expert_bias_update_policy must be one of "
                f"{tuple(sorted(valid_policies))}, got {expert_bias_update_policy!r}."
            )
            raise ValueError(msg)
        valid_schedules = {"constant", "linear"}
        if expert_bias_update_schedule not in valid_schedules:
            msg = (
                "expert_bias_update_schedule must be one of "
                f"{tuple(sorted(valid_schedules))}, got {expert_bias_update_schedule!r}."
            )
            raise ValueError(msg)
        if expert_bias_update_schedule == "linear":
            if expert_bias_update_schedule_steps is None:
                msg = "expert_bias_update_schedule_steps is required for linear bias update schedule."
                raise ValueError(msg)
            _validate_positive("expert_bias_update_schedule_steps", expert_bias_update_schedule_steps)
        if not 0.0 <= expert_bias_ema_beta < 1.0:
            msg = f"expert_bias_ema_beta must satisfy 0 <= beta < 1, got {expert_bias_ema_beta}."
            raise ValueError(msg)
        if not 0.0 <= expert_bias_adaptive_beta_min <= expert_bias_adaptive_beta_max < 1.0:
            msg = (
                "adaptive beta bounds must satisfy 0 <= min <= max < 1, got "
                f"{expert_bias_adaptive_beta_min} and {expert_bias_adaptive_beta_max}."
            )
            raise ValueError(msg)
        if expert_bias_adaptive_variance_reference <= 0.0:
            msg = (
                "expert_bias_adaptive_variance_reference must be greater than zero, "
                f"got {expert_bias_adaptive_variance_reference}."
            )
            raise ValueError(msg)
        if not 0.0 <= expert_bias_adaptive_state_decay < 1.0:
            msg = (
                "expert_bias_adaptive_state_decay must satisfy 0 <= decay < 1, "
                f"got {expert_bias_adaptive_state_decay}."
            )
            raise ValueError(msg)

        self.top_k = int(num_experts_per_tok)
        self.num_experts = int(num_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.hidden_dim = int(hidden_size)
        self.expert_bias_update_rate = float(expert_bias_update_rate)
        self.expert_bias_update_policy = expert_bias_update_policy
        self.expert_bias_update_interval = int(expert_bias_update_interval)
        self.expert_bias_ema_beta = float(expert_bias_ema_beta)
        self.expert_bias_adaptive_beta_min = float(expert_bias_adaptive_beta_min)
        self.expert_bias_adaptive_beta_max = float(expert_bias_adaptive_beta_max)
        self.expert_bias_adaptive_variance_reference = float(expert_bias_adaptive_variance_reference)
        self.expert_bias_adaptive_state_decay = float(expert_bias_adaptive_state_decay)
        self.expert_bias_update_topk = int(expert_bias_update_topk)
        self.expert_bias_update_schedule = expert_bias_update_schedule
        self.expert_bias_update_schedule_steps = (
            None if expert_bias_update_schedule_steps is None else int(expert_bias_update_schedule_steps)
        )
        self.expert_bias_update_end_rate = float(expert_bias_update_end_rate)
        self.expert_bias_clip = None if expert_bias_clip is None else float(expert_bias_clip)
        self.expert_bias_warmup_steps = int(expert_bias_warmup_steps)
        self.expert_bias_max_update_steps = (
            None if expert_bias_max_update_steps is None else int(expert_bias_max_update_steps)
        )

        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "expert_bias",
            torch.full((self.num_experts,), float(expert_bias_init), dtype=torch.float32),
        )
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("bias_update_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_expert_load", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("accumulated_expert_load", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("last_load_fraction", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("last_bias_delta", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("last_bias_update_rate", torch.zeros((), dtype=torch.float32))
        self.register_buffer("load_error_ema", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("load_error_accumulator", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("previous_load_error", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("adaptive_state_initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("persistent_energy_ema", torch.zeros((), dtype=torch.float32))
        self.register_buffer("oscillation_energy_ema", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "last_adaptive_ema_beta",
            torch.tensor(self.expert_bias_adaptive_beta_max, dtype=torch.float32),
        )
        self.register_buffer("last_normalized_load_variance", torch.zeros((), dtype=torch.float32))
        self.register_buffer("last_excess_load_variance", torch.zeros((), dtype=torch.float32))
        self.register_buffer("last_batch_noise", torch.zeros((), dtype=torch.float32))

    def _apply(self, fn: Any, recurse: bool = True) -> "Qwen3MoeAuxiliaryLossFreeTopKRouter":
        """Apply module conversions while keeping router control state in fp32."""

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

    @classmethod
    def from_qwen3_router(
        cls,
        router: nn.Module,
        *,
        expert_bias_init: float = 0.0,
        expert_bias_update_rate: float = 0.0,
        expert_bias_update_policy: str = "proportional",
        expert_bias_update_interval: int = 1,
        expert_bias_ema_beta: float = 0.9,
        expert_bias_adaptive_beta_min: float = 0.1,
        expert_bias_adaptive_beta_max: float = 0.95,
        expert_bias_adaptive_variance_reference: float = 2.5e-3,
        expert_bias_adaptive_state_decay: float = 0.9,
        expert_bias_update_topk: int = 1,
        expert_bias_update_schedule: str = "constant",
        expert_bias_update_schedule_steps: int | None = None,
        expert_bias_update_end_rate: float = 0.0,
        expert_bias_clip: float | None = None,
        expert_bias_warmup_steps: int = 0,
        expert_bias_max_update_steps: int | None = None,
    ) -> "Qwen3MoeAuxiliaryLossFreeTopKRouter":
        """Build an auxiliary-loss-free router from an existing Qwen3 router.

        Args:
            router: Existing router module with Qwen3-compatible attributes.
            expert_bias_init: Initial scalar value copied into all expert bias entries.
            expert_bias_update_rate: Update magnitude used for load-balancing bias steps.
            expert_bias_update_policy: Bias update policy.
            expert_bias_update_interval: Number of optimizer steps between updates.
            expert_bias_adaptive_beta_min: Minimum adaptive EMA beta.
            expert_bias_adaptive_beta_max: Maximum adaptive EMA beta.
            expert_bias_adaptive_variance_reference: Excess normalized load variance
                at the midpoint of the variance-adaptive beta mapping.
            expert_bias_adaptive_state_decay: Decay for persistent and oscillation
                energy estimates.
            expert_bias_update_topk: Number of positive-error and negative-error experts
                updated by the ``balanced_topk_sign`` policy.
            expert_bias_update_schedule: Schedule used for bias update rates.
            expert_bias_update_schedule_steps: Number of post-warmup optimizer steps
                used by the schedule.
            expert_bias_update_end_rate: Final bias update rate for scheduled decay.
            expert_bias_clip: Optional symmetric clip magnitude for bias entries.
            expert_bias_warmup_steps: Number of optimizer steps to skip before updates.
            expert_bias_max_update_steps: Optional last optimizer step allowed to
                update expert bias. ``None`` allows updates indefinitely.

        Returns:
            A new router initialized with copied Qwen3 router weights.

        Raises:
            AttributeError: If the source router does not expose required attributes.
        """

        replacement = cls(
            hidden_size=int(router.hidden_dim),
            num_experts=int(router.num_experts),
            num_experts_per_tok=int(router.top_k),
            norm_topk_prob=bool(router.norm_topk_prob),
            expert_bias_init=expert_bias_init,
            expert_bias_update_rate=expert_bias_update_rate,
            expert_bias_update_policy=expert_bias_update_policy,
            expert_bias_update_interval=expert_bias_update_interval,
            expert_bias_ema_beta=expert_bias_ema_beta,
            expert_bias_adaptive_beta_min=expert_bias_adaptive_beta_min,
            expert_bias_adaptive_beta_max=expert_bias_adaptive_beta_max,
            expert_bias_adaptive_variance_reference=expert_bias_adaptive_variance_reference,
            expert_bias_adaptive_state_decay=expert_bias_adaptive_state_decay,
            expert_bias_update_topk=expert_bias_update_topk,
            expert_bias_update_schedule=expert_bias_update_schedule,
            expert_bias_update_schedule_steps=expert_bias_update_schedule_steps,
            expert_bias_update_end_rate=expert_bias_update_end_rate,
            expert_bias_clip=expert_bias_clip,
            expert_bias_warmup_steps=expert_bias_warmup_steps,
            expert_bias_max_update_steps=expert_bias_max_update_steps,
        )
        replacement.to(device=router.weight.device, dtype=router.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(router.weight.detach())
        return replacement

    def _record_expert_load(self, router_indices: Tensor) -> None:
        """Record and accumulate load statistics for one routing decision.

        Args:
            router_indices: Selected expert indices with shape `(tokens, top_k)`.
        """

        with torch.no_grad():
            expert_load = torch.bincount(router_indices.reshape(-1), minlength=self.num_experts)
            expert_load = expert_load.to(device=self.last_expert_load.device, dtype=torch.long)
            if self.training and dist.is_available() and dist.is_initialized():
                dist.all_reduce(expert_load, op=dist.ReduceOp.SUM)
            self._set_load_statistics(expert_load)
            if self.training:
                self.accumulated_expert_load.add_(
                    expert_load.to(device=self.accumulated_expert_load.device, dtype=self.accumulated_expert_load.dtype)
                )

    def _set_load_statistics(self, expert_load: Tensor) -> None:
        """Store expert-count and fraction statistics.

        Args:
            expert_load: Per-expert assignment counts.
        """

        self.last_expert_load.copy_(expert_load.to(device=self.last_expert_load.device, dtype=torch.long))
        total_assignments = int(expert_load.sum().item())
        if total_assignments == 0:
            self.last_load_fraction.zero_()
            return
        load_fraction = expert_load.to(dtype=torch.float32) / float(total_assignments)
        self.last_load_fraction.copy_(load_fraction.to(device=self.last_load_fraction.device))

    def reset_expert_load_accumulator(self) -> None:
        """Reset expert load accumulated for the current optimizer step."""

        with torch.no_grad():
            self.accumulated_expert_load.zero_()

    def update_expert_bias_from_accumulated_load(self) -> bool:
        """Update expert bias once from accumulated optimizer-step load.

        Returns:
            `True` when the call produced a bias update event.
        """

        with torch.no_grad():
            accumulated_load = self.accumulated_expert_load.detach().clone()
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
        """Update the non-gradient expert bias from the latest observed load."""

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

            target_fraction = torch.full_like(self.last_load_fraction, 1.0 / float(self.num_experts))
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
                elif self.expert_bias_update_policy in {
                    "adaptive_ema_variance",
                    "adaptive_ema_persistent_oscillation",
                }:
                    self._update_adaptive_load_error_ema(load_error)
                    bias_delta = update_rate * self.load_error_ema
                else:
                    bias_delta = update_rate * load_error

            self.expert_bias.add_(bias_delta.to(device=self.expert_bias.device, dtype=self.expert_bias.dtype))
            if self.expert_bias_clip is not None:
                self.expert_bias.clamp_(-self.expert_bias_clip, self.expert_bias_clip)
            self.last_bias_delta.copy_(bias_delta.to(device=self.last_bias_delta.device, dtype=self.last_bias_delta.dtype))
            self.last_bias_update_rate.fill_(float(update_rate))
            self.bias_update_steps.add_(1)

    def _adaptive_load_statistics(self, load_error: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute batch-noise-corrected load statistics for adaptive EMA.

        Args:
            load_error: Current target-minus-observed expert load fractions.

        Returns:
            Normalized variance, excess variance, and finite-batch noise floor.
        """

        normalized_variance = float(self.num_experts) * load_error.square().sum()
        total_assignments = self.last_expert_load.sum().to(
            device=load_error.device, dtype=load_error.dtype
        )
        batch_noise = load_error.new_tensor(float(self.num_experts - 1)) / total_assignments.clamp_min(1.0)
        excess_variance = (normalized_variance - batch_noise).clamp_min(0.0)
        self.last_normalized_load_variance.copy_(normalized_variance)
        self.last_excess_load_variance.copy_(excess_variance)
        self.last_batch_noise.copy_(batch_noise)
        return normalized_variance, excess_variance, batch_noise

    def _update_adaptive_load_error_ema(self, load_error: Tensor) -> None:
        """Update load-error EMA with a beta computed from current routing state.

        Args:
            load_error: Current target-minus-observed expert load fractions.
        """

        load_error = load_error.to(device=self.load_error_ema.device, dtype=self.load_error_ema.dtype)
        _, excess_variance, batch_noise = self._adaptive_load_statistics(load_error)
        beta_range = self.expert_bias_adaptive_beta_max - self.expert_bias_adaptive_beta_min

        if self.expert_bias_update_policy == "adaptive_ema_variance":
            magnitude = excess_variance / (
                excess_variance + self.expert_bias_adaptive_variance_reference
            )
            beta = self.expert_bias_adaptive_beta_max - beta_range * magnitude
        else:
            initialized = self.adaptive_state_initialized
            previous_error = torch.where(initialized, self.previous_load_error, load_error)
            persistent = 0.5 * (load_error + previous_error)
            oscillatory = 0.5 * (load_error - previous_error)
            persistent_energy = float(self.num_experts) * persistent.square().sum()
            oscillation_energy = float(self.num_experts) * oscillatory.square().sum()
            decay = self.expert_bias_adaptive_state_decay
            smoothed_persistent = torch.where(
                initialized,
                decay * self.persistent_energy_ema + (1.0 - decay) * persistent_energy,
                persistent_energy,
            )
            smoothed_oscillation = torch.where(
                initialized,
                decay * self.oscillation_energy_ema + (1.0 - decay) * oscillation_energy,
                oscillation_energy,
            )
            self.persistent_energy_ema.copy_(smoothed_persistent)
            self.oscillation_energy_ema.copy_(smoothed_oscillation)
            beta = (smoothed_oscillation + batch_noise) / (
                smoothed_persistent + smoothed_oscillation + batch_noise
            )

        beta = beta.clamp(
            min=self.expert_bias_adaptive_beta_min,
            max=self.expert_bias_adaptive_beta_max,
        )
        self.load_error_ema.mul_(beta).add_(load_error * (1.0 - beta))
        self.previous_load_error.copy_(load_error)
        self.adaptive_state_initialized.fill_(True)
        self.last_adaptive_ema_beta.copy_(beta)

    def _scheduled_bias_update_rate(self, steps_after_warmup: int) -> float:
        """Compute the bias update rate for the current post-warmup step.

        Args:
            steps_after_warmup: One-indexed optimizer-step count after warmup.

        Returns:
            The scalar bias update rate to apply for this step.
        """

        if self.expert_bias_update_schedule == "constant":
            return self.expert_bias_update_rate

        if self.expert_bias_update_schedule == "linear":
            schedule_steps = int(self.expert_bias_update_schedule_steps or 1)
            if schedule_steps <= 1:
                return self.expert_bias_update_end_rate
            progress = min(max((steps_after_warmup - 1) / float(schedule_steps - 1), 0.0), 1.0)
            return self.expert_bias_update_rate + progress * (
                self.expert_bias_update_end_rate - self.expert_bias_update_rate
            )

        msg = f"Unsupported expert_bias_update_schedule: {self.expert_bias_update_schedule!r}."
        raise ValueError(msg)

    def _balanced_topk_sign(self, load_error: Tensor) -> Tensor:
        """Select equal-size positive and negative top-k sign updates."""

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

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route hidden states with bias-based selection and probability-based weights.

        Args:
            hidden_states: Input hidden states of shape `(..., hidden_size)`.

        Returns:
            A tuple of `(router_logits, router_scores, router_indices)` matching the
            Hugging Face Qwen3 MoE router forward contract.
        """

        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
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

        self._record_expert_load(router_indices)

        return router_logits, router_scores, router_indices


def router_hyperparameters_from_config(config: Any) -> dict[str, Any]:
    """Extract Qwen3 MoE router hyperparameters from a config object.

    Args:
        config: Config object with Qwen3 MoE router attributes.

    Returns:
        A dictionary of router construction keyword arguments.
    """

    return {
        "hidden_size": int(config.hidden_size),
        "num_experts": int(config.num_experts),
        "num_experts_per_tok": int(config.num_experts_per_tok),
        "norm_topk_prob": bool(config.norm_topk_prob),
    }


AuxLossFreeRouter = Qwen3MoeAuxiliaryLossFreeTopKRouter

__all__ = [
    "AuxLossFreeRouter",
    "Qwen3MoeAuxiliaryLossFreeTopKRouter",
    "router_hyperparameters_from_config",
]
