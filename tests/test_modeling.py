"""Unit tests for Qwen3 MoE model patching helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alf.config import AlfConfig, ModelConfig
from alf.metrics import collect_auxiliary_loss_free_router_metrics
from alf.modeling import (
    _record_plain_router_load_hook,
    apply_aux_loss_free_router,
    build_model_and_tokenizer,
    iter_auxiliary_loss_free_routers,
    replace_qwen3_moe_routers,
)
from alf.router import Qwen3MoeAuxiliaryLossFreeTopKRouter


class FakeSparseBlock(nn.Module):
    """Hold a Qwen3-compatible router for patching tests.

    Attributes:
        gate: Router module under test.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        """Initialize the fake sparse block.

        Args:
            config: Qwen3 MoE configuration used to build the router.
        """

        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)


class FakeQwen3MoeModel(nn.Module):
    """Provide a minimal nested model for router replacement tests.

    Attributes:
        config: Model config carrying the router auxiliary-loss coefficient.
        router_aux_loss_coef: Mirrored attribute used by Hugging Face causal LM classes.
        blocks: Nested module list containing sparse blocks.
    """

    def __init__(self) -> None:
        """Initialize the fake Qwen3 MoE model."""

        super().__init__()
        self.config = Qwen3MoeConfig(
            hidden_size=4,
            intermediate_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_experts=3,
            num_experts_per_tok=2,
            vocab_size=32,
            output_router_logits=True,
            router_aux_loss_coef=0.5,
        )
        self.router_aux_loss_coef = 0.5
        self.blocks = nn.ModuleList([FakeSparseBlock(self.config), FakeSparseBlock(self.config)])


@dataclass(frozen=True)
class FakeAlfConfig:
    """ALF config object used by apply_aux_loss_free_router tests.

    Attributes:
        enabled: Whether ALF router patching is enabled.
        bias_init: Initial expert-bias value.
        bias_update_rate: Bias update rate.
        bias_update_policy: Bias update policy.
        bias_adaptive_beta_min: Minimum adaptive EMA beta.
        bias_adaptive_beta_max: Maximum adaptive EMA beta.
        bias_adaptive_variance_reference: Variance-adaptive mapping midpoint.
        bias_adaptive_state_decay: Persistent/oscillation energy decay.
        bias_gain_coupled_normalized_gain: Stability-normalized feedback gain.
        bias_gain_coupled_rate_min: Minimum gain-coupled update rate.
        bias_gain_coupled_rate_max: Maximum gain-coupled update rate.
        bias_adaptive_per_expert_beta: EMA decay for per-expert squared load error.
        bias_adaptive_per_expert_momentum_beta: EMA decay for per-expert
            load-error momentum.
        bias_adaptive_per_expert_epsilon: Adaptive-rate denominator stabilizer.
        update_interval: Number of forwards between bias updates.
        bias_update_topk: Number of positive-error and negative-error experts
            updated by balanced top-k sign.
        bias_clip: Optional absolute bias clipping value.
        warmup_steps: Number of optimizer steps before updates begin.
        bias_max_update_steps: Optional last optimizer step allowed to update bias.
        disable_router_aux_loss: Whether to zero Qwen's router aux-loss coefficient.
    """

    enabled: bool = True
    bias_init: float = 0.125
    bias_update_rate: float = 0.05
    bias_update_policy: str = "sign"
    bias_adaptive_beta_min: float = 0.2
    bias_adaptive_beta_max: float = 0.8
    bias_adaptive_variance_reference: float = 0.01
    bias_adaptive_state_decay: float = 0.7
    bias_gain_coupled_normalized_gain: float = 0.04
    bias_gain_coupled_rate_min: float = 0.06
    bias_gain_coupled_rate_max: float = 0.2
    bias_adaptive_per_expert_beta: float = 0.65
    bias_adaptive_per_expert_momentum_beta: float = 0.7
    bias_adaptive_per_expert_epsilon: float = 1e-6
    update_interval: int = 4
    bias_update_topk: int = 2
    bias_clip: float | None = 0.75
    warmup_steps: int = 2
    bias_max_update_steps: int | None = 5
    disable_router_aux_loss: bool = False


def test_replace_qwen3_moe_routers_copies_router_weights_and_disables_aux_loss() -> None:
    """Router replacement should preserve weights and disable auxiliary loss."""

    model = FakeQwen3MoeModel()
    original_weights = []
    for index, block in enumerate(model.blocks):
        with torch.no_grad():
            block.gate.weight.copy_(torch.full_like(block.gate.weight, float(index + 1)))
        original_weights.append(block.gate.weight.detach().clone())

    summary = replace_qwen3_moe_routers(
        model,
        expert_bias_init=0.25,
        expert_bias_update_rate=0.1,
        expert_bias_update_interval=2,
        expert_bias_adaptive_beta_min=0.15,
        expert_bias_adaptive_beta_max=0.85,
        expert_bias_adaptive_variance_reference=0.02,
        expert_bias_adaptive_state_decay=0.6,
        expert_bias_gain_coupled_normalized_gain=0.03,
        expert_bias_gain_coupled_rate_min=0.04,
        expert_bias_gain_coupled_rate_max=0.25,
        expert_bias_adaptive_per_expert_beta=0.75,
        expert_bias_adaptive_per_expert_momentum_beta=0.8,
        expert_bias_adaptive_per_expert_epsilon=1e-7,
        expert_bias_clip=0.5,
        expert_bias_warmup_steps=3,
        expert_bias_max_update_steps=6,
    )

    routers = list(iter_auxiliary_loss_free_routers(model))

    assert summary["num_replaced"] == 2
    assert summary["router_aux_loss_disabled"] is True
    assert summary["replaced_router_paths"] == ["blocks.0.gate", "blocks.1.gate"]
    assert model.router_aux_loss_coef == 0.0
    assert model.config.router_aux_loss_coef == 0.0
    assert len(routers) == 2

    for (name, router), expected_weight in zip(routers, original_weights, strict=True):
        assert name.endswith(".gate")
        assert isinstance(router, Qwen3MoeAuxiliaryLossFreeTopKRouter)
        assert torch.allclose(router.weight, expected_weight)
        assert torch.allclose(router.expert_bias, torch.full((3,), 0.25))
        assert router.expert_bias_update_rate == 0.1
        assert router.expert_bias_update_interval == 2
        assert router.expert_bias_adaptive_beta_min == 0.15
        assert router.expert_bias_adaptive_beta_max == 0.85
        assert router.expert_bias_adaptive_variance_reference == 0.02
        assert router.expert_bias_adaptive_state_decay == 0.6
        assert router.expert_bias_gain_coupled_normalized_gain == 0.03
        assert router.expert_bias_gain_coupled_rate_min == 0.04
        assert router.expert_bias_gain_coupled_rate_max == 0.25
        assert router.expert_bias_adaptive_per_expert_beta == 0.75
        assert router.expert_bias_adaptive_per_expert_momentum_beta == 0.8
        assert router.expert_bias_adaptive_per_expert_epsilon == 1e-7
        assert router.expert_bias_clip == 0.5
        assert router.expert_bias_warmup_steps == 3
        assert router.expert_bias_max_update_steps == 6


def test_apply_aux_loss_free_router_reads_alf_config_object() -> None:
    """The requested patching utility should adapt project AlfConfig fields."""

    model = FakeQwen3MoeModel()

    summary = apply_aux_loss_free_router(model, FakeAlfConfig())
    routers = list(iter_auxiliary_loss_free_routers(model))

    assert summary["num_replaced"] == 2
    assert summary["patched_routers"] == 2
    assert summary["patched_module_names"] == ("blocks.0.gate", "blocks.1.gate")
    assert summary["router_aux_loss_disabled"] is False
    assert model.router_aux_loss_coef == 0.5
    assert model.config.router_aux_loss_coef == 0.5
    assert len(routers) == 2

    for _, router in routers:
        assert torch.allclose(router.expert_bias, torch.full((3,), 0.125))
        assert router.expert_bias_update_rate == 0.05
        assert router.expert_bias_update_policy == "sign"
        assert router.expert_bias_update_interval == 4
        assert router.expert_bias_adaptive_beta_min == 0.2
        assert router.expert_bias_adaptive_beta_max == 0.8
        assert router.expert_bias_adaptive_variance_reference == 0.01
        assert router.expert_bias_adaptive_state_decay == 0.7
        assert router.expert_bias_gain_coupled_normalized_gain == 0.04
        assert router.expert_bias_gain_coupled_rate_min == 0.06
        assert router.expert_bias_gain_coupled_rate_max == 0.2
        assert router.expert_bias_adaptive_per_expert_beta == 0.65
        assert router.expert_bias_adaptive_per_expert_momentum_beta == 0.7
        assert router.expert_bias_adaptive_per_expert_epsilon == 1e-6
        assert router.expert_bias_update_topk == 2
        assert router.expert_bias_clip == 0.75
        assert router.expert_bias_warmup_steps == 2
        assert router.expert_bias_max_update_steps == 5


def test_collect_auxiliary_loss_free_router_metrics_aggregates_serializable_values() -> None:
    """Metric collection should aggregate counts and bias summaries across routers."""

    model = FakeQwen3MoeModel()
    replace_qwen3_moe_routers(model, expert_bias_update_rate=0.0)

    first_router = model.blocks[0].gate
    second_router = model.blocks[1].gate
    assert isinstance(first_router, Qwen3MoeAuxiliaryLossFreeTopKRouter)
    assert isinstance(second_router, Qwen3MoeAuxiliaryLossFreeTopKRouter)

    with torch.no_grad():
        first_router.weight.copy_(torch.tensor([[4.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
        first_router.expert_bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
        second_router.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]))
        second_router.expert_bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

    hidden_states = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    first_router(hidden_states)
    second_router(hidden_states)

    metrics = collect_auxiliary_loss_free_router_metrics(model)

    assert metrics["num_routers"] == 2
    assert set(metrics["routers"]) == {"blocks.0.gate", "blocks.1.gate"}
    assert metrics["aggregate_load"]["counts"] == [2, 1, 1]
    assert metrics["aggregate_load"]["max_min_load_ratio"] == 2.0
    assert len(metrics["aggregate_bias"]["values"]) == 6


def test_build_model_uses_configured_router_aux_loss_for_aux_baseline() -> None:
    """Aux-loss baseline should read router aux-loss coefficient from ModelConfig."""

    model, _ = build_model_and_tokenizer(
        ModelConfig(router_aux_loss_coef=0.02),
        AlfConfig(enabled=False, disable_router_aux_loss=False),
    )

    assert model.router_aux_loss_coef == 0.02
    assert model.config.router_aux_loss_coef == 0.02


def test_build_model_disables_configured_router_aux_loss_for_alf() -> None:
    """ALF should still disable configured router aux loss when requested."""

    model, _ = build_model_and_tokenizer(
        ModelConfig(router_aux_loss_coef=0.02),
        AlfConfig(enabled=True, disable_router_aux_loss=True),
    )

    assert model.router_aux_loss_coef == 0.0
    assert model.config.router_aux_loss_coef == 0.0


def test_plain_router_load_hook_all_reduces_during_ddp_training(monkeypatch) -> None:
    """Aux-loss baseline load metrics should use global DDP counts during training."""

    class FakeDist:
        """Minimal torch.distributed test double for all-reduce behavior."""

        class ReduceOp:
            """Reduction operation names used by the hook."""

            SUM = "sum"

        def __init__(self) -> None:
            """Initialize the fake distributed backend."""

            self.calls = 0

        def is_available(self) -> bool:
            """Return whether distributed collectives are available."""

            return True

        def is_initialized(self) -> bool:
            """Return whether a process group is initialized."""

            return True

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            """Simulate a sum all-reduce by doubling the local counts."""

            assert op == self.ReduceOp.SUM
            self.calls += 1
            tensor.mul_(2)

    fake_dist = FakeDist()
    monkeypatch.setattr("alf.modeling.dist", fake_dist)
    module = nn.Module()
    module.num_experts = 3
    module.register_buffer("last_expert_load", torch.zeros(3, dtype=torch.long))
    module.register_buffer("last_load_fraction", torch.zeros(3, dtype=torch.float32))
    module.train()

    router_indices = torch.tensor([[0, 1], [1, 2]])
    _record_plain_router_load_hook(module, (), (torch.empty(0), torch.empty(0), router_indices))

    assert fake_dist.calls == 1
    assert module.last_expert_load.tolist() == [2, 4, 2]
    assert torch.allclose(module.last_load_fraction, torch.tensor([0.25, 0.5, 0.25]))


def test_plain_router_load_hook_skips_all_reduce_during_eval(monkeypatch) -> None:
    """Rank-zero-only eval should not enter a distributed collective."""

    class FakeDist:
        """Distributed test double that fails if all-reduce is called."""

        class ReduceOp:
            """Reduction operation names used by the hook."""

            SUM = "sum"

        def is_available(self) -> bool:
            """Return whether distributed collectives are available."""

            return True

        def is_initialized(self) -> bool:
            """Return whether a process group is initialized."""

            return True

        def all_reduce(self, tensor: torch.Tensor, op: str) -> None:
            """Fail if eval tries to synchronize router metrics."""

            raise AssertionError("eval hook should not all_reduce")

    monkeypatch.setattr("alf.modeling.dist", FakeDist())
    module = nn.Module()
    module.num_experts = 2
    module.register_buffer("last_expert_load", torch.zeros(2, dtype=torch.long))
    module.register_buffer("last_load_fraction", torch.zeros(2, dtype=torch.float32))
    module.eval()

    router_indices = torch.tensor([[0, 1]])
    _record_plain_router_load_hook(module, (), (torch.empty(0), torch.empty(0), router_indices))

    assert module.last_expert_load.tolist() == [1, 1]
