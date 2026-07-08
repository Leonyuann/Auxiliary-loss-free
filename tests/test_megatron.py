"""Tests for Megatron Core experiment configuration and router helpers."""

from __future__ import annotations

import torch

from alf.config import load_experiment_config
from alf.megatron_router import MegatronAuxiliaryLossFreeTopKRouter, reduce_expert_load_counts
from alf.megatron_train import (
    estimate_moe_total_parameters,
    megatron_parallel_world_size,
    megatron_transformer_config_kwargs,
    validate_megatron_config,
)


def test_c4_1b_megatron_configs_use_ep4_dp2_top3_defaults() -> None:
    """Megatron 1B configs should encode the requested 8xA100 parallelism."""

    for path in [
        "experiments/qwen3_moe_c4_1b_megatron_alf.py",
        "experiments/qwen3_moe_c4_1b_megatron_alf_ema.py",
        "experiments/qwen3_moe_c4_1b_megatron_aux_loss.py",
    ]:
        config = load_experiment_config(path)

        assert config.megatron.enabled is True
        assert config.megatron.tensor_model_parallel_size == 1
        assert config.megatron.pipeline_model_parallel_size == 1
        assert config.megatron.context_parallel_size == 1
        assert config.megatron.expert_model_parallel_size == 4
        assert config.megatron.data_parallel_size == 2
        assert megatron_parallel_world_size(config) == 8
        assert config.model.num_experts == 24
        assert config.model.num_experts_per_tok == 3
        assert config.model.num_experts // config.megatron.expert_model_parallel_size == 6
        validate_megatron_config(config)


def test_c4_1b_megatron_shape_is_about_one_billion_parameters() -> None:
    """The default Megatron shape should be clearly larger than the 300M DDP configs."""

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")

    parameter_count = estimate_moe_total_parameters(config)

    assert 1_000_000_000 <= parameter_count <= 1_200_000_000


def test_megatron_transformer_kwargs_disable_aux_loss_for_alf_only() -> None:
    """ALF should disable Megatron router aux loss while aux baseline keeps it."""

    alf_config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    aux_config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_aux_loss.py")

    alf_kwargs = megatron_transformer_config_kwargs(alf_config)
    aux_kwargs = megatron_transformer_config_kwargs(aux_config)

    assert alf_kwargs["num_moe_experts"] == 24
    assert alf_kwargs["moe_router_topk"] == 3
    assert alf_kwargs["moe_aux_loss_coeff"] == 0.0
    assert aux_kwargs["moe_aux_loss_coeff"] == aux_config.model.router_aux_loss_coef


def test_megatron_alf_router_returns_top3_probability_map() -> None:
    """Megatron ALF router should expose dense top3 probabilities and routing map."""

    router = MegatronAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=4,
        num_experts_per_tok=3,
        norm_topk_prob=False,
        expert_bias_update_rate=0.0,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))

    probs, routing_map = router(torch.tensor([[1.0, 0.0]]))

    raw_probs = torch.softmax(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), dim=-1)
    assert probs.shape == (1, 4)
    assert routing_map.shape == (1, 4)
    assert routing_map.tolist() == [[True, True, False, True]]
    assert torch.allclose(probs[0, torch.tensor([3, 0, 1])], raw_probs[0, torch.tensor([3, 0, 1])])
    assert probs[0, 2] == 0
    assert router.last_expert_load.tolist() == [1, 1, 0, 1]


def test_megatron_alf_router_padding_mask_excludes_load() -> None:
    """Padding tokens should not contribute to optimizer-step ALF bias updates."""

    router = MegatronAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=3,
        num_experts_per_tok=3,
        norm_topk_prob=False,
        expert_bias_update_rate=0.0,
    )
    with torch.no_grad():
        router.weight.zero_()

    router.train()
    _, routing_map = router(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        padding_mask=torch.tensor([False, True]),
    )

    assert routing_map[0].sum().item() == 3
    assert routing_map[1].sum().item() == 0
    assert router.accumulated_expert_load.sum().item() == 3


def test_reduce_expert_load_counts_only_uses_explicit_group(monkeypatch) -> None:
    """Load reduction should be opt-in so EP ranks are not reduced by accident."""

    class FakeReduceOp:
        """Fake torch distributed reduce operation namespace."""

        SUM = "sum"

    class FakeDist:
        """Fake distributed module that marks explicit all-reduce calls."""

        ReduceOp = FakeReduceOp

        @staticmethod
        def is_available() -> bool:
            """Return that distributed collectives are available."""

            return True

        @staticmethod
        def is_initialized() -> bool:
            """Return that distributed collectives are initialized."""

            return True

        @staticmethod
        def all_reduce(tensor: torch.Tensor, op: str, group: object | None = None) -> None:
            """Mark reductions by adding a deterministic offset."""

            assert op == "sum"
            assert group == "tp_cp_dp"
            tensor.add_(10)

    import alf.megatron_router as megatron_router

    monkeypatch.setattr(megatron_router, "dist", FakeDist)

    local = torch.tensor([1, 2, 3])

    assert reduce_expert_load_counts(local, reduce_group=None).tolist() == [1, 2, 3]
    assert reduce_expert_load_counts(local, reduce_group="tp_cp_dp").tolist() == [11, 12, 13]
