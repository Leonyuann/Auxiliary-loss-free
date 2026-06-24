"""Auxiliary-loss-free Qwen3 MoE router implementations."""

from __future__ import annotations

from typing import Any

import torch
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
        expert_bias_update_interval: Number of training forwards between bias updates.
        expert_bias_clip: Optional absolute clipping value for expert bias entries.
        expert_bias_warmup_steps: Number of training forwards to skip before updates.
        weight: Router projection weights with shape `(num_experts, hidden_dim)`.
        expert_bias: Non-gradient bias applied only for top-k expert selection.
        training_steps: Number of training-mode forwards seen by this router.
        bias_update_steps: Number of times the expert bias has been updated.
        last_expert_load: Last observed selected-expert counts.
        last_load_fraction: Last observed selected-expert fractions.
        last_bias_delta: Last applied expert-bias update delta.
    """

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
        expert_bias_clip: float | None = None,
        expert_bias_warmup_steps: int = 0,
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
                `proportional` and `sign`.
            expert_bias_update_interval: Number of training forwards between updates.
            expert_bias_clip: Optional symmetric clip magnitude for bias entries.
            expert_bias_warmup_steps: Number of training forwards to skip before updates.

        Raises:
            ValueError: If any hyperparameter is inconsistent.
        """

        super().__init__()
        _validate_positive("hidden_size", hidden_size)
        _validate_positive("num_experts", num_experts)
        _validate_positive("num_experts_per_tok", num_experts_per_tok)
        _validate_positive("expert_bias_update_interval", expert_bias_update_interval)
        if num_experts_per_tok > num_experts:
            msg = (
                "num_experts_per_tok must be less than or equal to num_experts, "
                f"got {num_experts_per_tok} and {num_experts}."
            )
            raise ValueError(msg)
        if expert_bias_warmup_steps < 0:
            msg = f"expert_bias_warmup_steps must be non-negative, got {expert_bias_warmup_steps}."
            raise ValueError(msg)
        if expert_bias_clip is not None and expert_bias_clip < 0.0:
            msg = f"expert_bias_clip must be non-negative, got {expert_bias_clip}."
            raise ValueError(msg)
        if expert_bias_update_policy not in {"proportional", "sign"}:
            msg = (
                "expert_bias_update_policy must be one of "
                f"{('proportional', 'sign')}, got {expert_bias_update_policy!r}."
            )
            raise ValueError(msg)

        self.top_k = int(num_experts_per_tok)
        self.num_experts = int(num_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.hidden_dim = int(hidden_size)
        self.expert_bias_update_rate = float(expert_bias_update_rate)
        self.expert_bias_update_policy = expert_bias_update_policy
        self.expert_bias_update_interval = int(expert_bias_update_interval)
        self.expert_bias_clip = None if expert_bias_clip is None else float(expert_bias_clip)
        self.expert_bias_warmup_steps = int(expert_bias_warmup_steps)

        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "expert_bias",
            torch.full((self.num_experts,), float(expert_bias_init), dtype=torch.float32),
        )
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("bias_update_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_expert_load", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("last_load_fraction", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("last_bias_delta", torch.zeros(self.num_experts, dtype=torch.float32))

    @classmethod
    def from_qwen3_router(
        cls,
        router: nn.Module,
        *,
        expert_bias_init: float = 0.0,
        expert_bias_update_rate: float = 0.0,
        expert_bias_update_policy: str = "proportional",
        expert_bias_update_interval: int = 1,
        expert_bias_clip: float | None = None,
        expert_bias_warmup_steps: int = 0,
    ) -> "Qwen3MoeAuxiliaryLossFreeTopKRouter":
        """Build an auxiliary-loss-free router from an existing Qwen3 router.

        Args:
            router: Existing router module with Qwen3-compatible attributes.
            expert_bias_init: Initial scalar value copied into all expert bias entries.
            expert_bias_update_rate: Update magnitude used for load-balancing bias steps.
            expert_bias_update_policy: Bias update policy.
            expert_bias_update_interval: Number of training forwards between updates.
            expert_bias_clip: Optional symmetric clip magnitude for bias entries.
            expert_bias_warmup_steps: Number of training forwards to skip before updates.

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
            expert_bias_clip=expert_bias_clip,
            expert_bias_warmup_steps=expert_bias_warmup_steps,
        )
        replacement.to(device=router.weight.device, dtype=router.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(router.weight.detach())
        return replacement

    def _record_expert_load(self, router_indices: Tensor) -> None:
        """Record load statistics for the most recent routing decision.

        Args:
            router_indices: Selected expert indices with shape `(tokens, top_k)`.
        """

        with torch.no_grad():
            expert_load = torch.bincount(router_indices.reshape(-1), minlength=self.num_experts)
            self.last_expert_load.copy_(expert_load.to(device=self.last_expert_load.device, dtype=torch.long))
            total_assignments = int(expert_load.sum().item())
            if total_assignments == 0:
                self.last_load_fraction.zero_()
                return
            load_fraction = expert_load.to(dtype=torch.float32) / float(total_assignments)
            self.last_load_fraction.copy_(load_fraction.to(device=self.last_load_fraction.device))

    def _update_expert_bias(self) -> None:
        """Update the non-gradient expert bias from the latest observed load."""

        with torch.no_grad():
            self.training_steps.add_(1)
            self.last_bias_delta.zero_()

            if self.expert_bias_update_rate == 0.0:
                return

            current_step = int(self.training_steps.item())
            if current_step <= self.expert_bias_warmup_steps:
                return

            steps_after_warmup = current_step - self.expert_bias_warmup_steps
            if steps_after_warmup % self.expert_bias_update_interval != 0:
                return

            target_fraction = torch.full_like(self.last_load_fraction, 1.0 / float(self.num_experts))
            load_error = target_fraction - self.last_load_fraction
            if self.expert_bias_update_policy == "sign":
                bias_delta = self.expert_bias_update_rate * load_error.sign()
            else:
                bias_delta = self.expert_bias_update_rate * load_error
            self.expert_bias.add_(bias_delta.to(device=self.expert_bias.device, dtype=self.expert_bias.dtype))
            if self.expert_bias_clip is not None:
                self.expert_bias.clamp_(-self.expert_bias_clip, self.expert_bias_clip)
            self.last_bias_delta.copy_(bias_delta.to(device=self.last_bias_delta.device))
            self.bias_update_steps.add_(1)

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
        if self.training:
            self._update_expert_bias()

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
