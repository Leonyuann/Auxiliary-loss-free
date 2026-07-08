"""Tests for Megatron Core experiment configuration and router helpers."""

from __future__ import annotations

import torch

from types import SimpleNamespace

from alf.config import AlfConfig, ExperimentConfig, MegatronConfig, ModelConfig, load_experiment_config
from alf.megatron_router import (
    MegatronAuxiliaryLossFreeTopKRouter,
    MegatronCoreAuxiliaryLossFreeTopKRouter,
    reduce_expert_load_counts,
)
from alf.megatron_train import (
    estimate_moe_total_parameters,
    build_megatron_layer_spec,
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


def test_megatron_transformer_config_instantiates_for_alf_softmax_bias() -> None:
    """ALF kwargs should not use Megatron's invalid native softmax expert-bias mode."""

    from megatron.core.transformer.transformer_config import TransformerConfig

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    kwargs = megatron_transformer_config_kwargs(config)

    transformer_config = TransformerConfig(**kwargs)

    assert transformer_config.moe_router_enable_expert_bias is False
    assert transformer_config.moe_router_score_function == "softmax"


def test_megatron_layer_spec_installs_project_alf_router() -> None:
    """ALF Megatron models should use the project router, not native TopKRouter."""

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf_ema.py")

    layer_spec = build_megatron_layer_spec(config)
    router_factory = layer_spec.submodules.mlp.keywords["submodules"].router

    assert router_factory.func is MegatronCoreAuxiliaryLossFreeTopKRouter
    assert router_factory.keywords["alf_config"].bias_update_policy == "ema"
    assert router_factory.keywords["alf_config"].bias_ema_beta == 0.5


def test_megatron_core_alf_router_uses_accumulated_ema_update() -> None:
    """Core Megatron ALF router should preserve optimizer-step EMA semantics."""

    from megatron.core.transformer.transformer_config import TransformerConfig

    config = ExperimentConfig(
        name="tiny-megatron-router",
        model=ModelConfig(
            hidden_size=2,
            intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=3,
        ),
        megatron=MegatronConfig(enabled=True, expert_model_parallel_size=4, data_parallel_size=2),
        alf=AlfConfig(enabled=True, bias_update_rate=1.0, bias_update_policy="ema", bias_ema_beta=0.5),
    )
    kwargs = megatron_transformer_config_kwargs(config)
    kwargs["perform_initialization"] = False
    transformer_config = TransformerConfig(**kwargs)
    process_groups = SimpleNamespace(tp=None, cp=None, tp_cp=None, tp_dp_cp=None)
    router = MegatronCoreAuxiliaryLossFreeTopKRouter(
        config=transformer_config,
        pg_collection=process_groups,
        alf_config=config.alf,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))

    router.train()
    router.routing(torch.tensor([[[4.0, 3.0, 2.0, 1.0]]], device=router.weight.device))

    assert router.accumulated_expert_load.cpu().tolist() == [1, 1, 0, 1]
    assert router.update_expert_bias_from_accumulated_load() is True
    expected_ema = torch.tensor([-1 / 24, -1 / 24, 1 / 8, -1 / 24], device=router.load_error_ema.device)
    assert torch.allclose(router.load_error_ema, expected_ema)
    assert torch.allclose(router.last_bias_delta, router.load_error_ema)


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
