"""Tests for Megatron Core experiment configuration and router helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from alf.config import AlfConfig, ExperimentConfig, MegatronConfig, ModelConfig, load_experiment_config
from alf.megatron_router import (
    MegatronAuxiliaryLossFreeTopKRouter,
    MegatronCoreAuxiliaryLossFreeTopKRouter,
    reduce_expert_load_counts,
    update_megatron_alf_router_biases,
)
from alf.megatron_train import (
    estimate_moe_total_parameters,
    _build_megatron_sampler,
    _build_megatron_training_optimizer,
    _build_megatron_training_scheduler,
    _collect_megatron_load_observers,
    _install_megatron_load_observers,
    _load_megatron_rank_checkpoint,
    _megatron_backward_scale,
    _resolve_megatron_resume_checkpoint,
    _save_megatron_rank_checkpoint,
    _megatron_clip_grad,
    _megatron_global_tokens,
    _post_megatron_optimizer_step,
    _set_megatron_aux_loss_scale,
    _seed_megatron_model_parallel_rng,
    _step_megatron_optimizer,
    build_megatron_layer_spec,
    megatron_effective_global_batch_size,
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
        assert config.megatron.transformer_impl == "transformer_engine"
        assert config.megatron.moe_grouped_gemm is True
        assert config.megatron.overlap_grad_reduce is True
        assert config.megatron.overlap_param_gather is True
        assert megatron_parallel_world_size(config) == 8
        assert config.model.num_experts == 24
        assert config.model.num_experts_per_tok == 3
        assert config.model.num_experts // config.megatron.expert_model_parallel_size == 6
        assert megatron_effective_global_batch_size(config) == config.megatron.global_batch_size == 16
        validate_megatron_config(config)


def test_megatron_launch_uses_distinct_auto_resume_directories() -> None:
    """Launch branches should use isolated output dirs and implicit latest resume."""

    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_c4_1b_megatron_8xa100.sh"
    ).read_text(encoding="utf-8")

    assert 'output_root="${OUTPUT_ROOT:-${OUTPUT_DIR:-$project_root/outputs}}"' in script
    assert 'alf_output_dir="$output_root/qwen3_moe_c4_1b_megatron_alf"' in script
    assert 'ema_output_dir="$output_root/qwen3_moe_c4_1b_megatron_alf_ema"' in script
    assert 'aux_output_dir="$output_root/qwen3_moe_c4_1b_megatron_aux_loss"' in script
    assert '--training.output_dir "$alf_output_dir"' in script
    assert '--training.output_dir "$ema_output_dir"' in script
    assert '--training.output_dir "$aux_output_dir"' in script
    assert "--training.resume_from" not in script


def test_megatron_config_allows_consistent_two_gpu_smoke_topology(monkeypatch) -> None:
    """Default 8-GPU configs should permit smaller consistent smoke topologies."""

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    config.megatron.expert_model_parallel_size = 2
    config.megatron.data_parallel_size = 1
    config.megatron.global_batch_size = 8

    assert megatron_parallel_world_size(config) == 2
    validate_megatron_config(config)


def test_megatron_train_binds_device_and_seeds_rng_before_model_build(monkeypatch, tmp_path) -> None:
    """Training should bind CUDA and seed Megatron RNG before constructing GPT."""

    import alf.megatron_train as megatron_train

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    config.training.output_dir = str(tmp_path)
    events = []
    device = torch.device("cuda", 1)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(megatron_train, "load_experiment_config", lambda *args: config)
    monkeypatch.setattr(megatron_train, "_require_megatron_core", lambda: None)
    monkeypatch.setattr(megatron_train, "_resolve_megatron_device", lambda: events.append("device") or device)
    monkeypatch.setattr(
        megatron_train,
        "_init_torch_distributed_if_needed",
        lambda loaded, resolved: events.append(("distributed", resolved)),
    )
    monkeypatch.setattr(megatron_train, "_init_megatron_model_parallel", lambda loaded: events.append("parallel"))
    monkeypatch.setattr(megatron_train, "_seed_megatron_model_parallel_rng", lambda loaded: events.append("rng"))
    monkeypatch.setattr(megatron_train, "_is_global_rank_zero", lambda: False)
    monkeypatch.setattr(
        megatron_train,
        "_run_megatron_training_loop",
        lambda loaded, output, resolved: events.append(("model", resolved)),
    )
    monkeypatch.setattr(megatron_train, "_cleanup_megatron_model_parallel", lambda: None)
    monkeypatch.setattr(megatron_train, "_cleanup_torch_distributed_if_needed", lambda: None)

    megatron_train.train("unused.py")

    assert events == [
        "device",
        ("distributed", device),
        "parallel",
        "rng",
        ("model", device),
    ]


def test_megatron_cuda_rng_seed_uses_training_seed(monkeypatch) -> None:
    """Megatron model-parallel CUDA RNG should receive the experiment seed."""

    import megatron.core.tensor_parallel.random as megatron_random

    seeds = []
    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    config.training.seed = 2468
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_random, "model_parallel_cuda_manual_seed", seeds.append)

    _seed_megatron_model_parallel_rng(config)

    assert seeds == [2468]


def test_megatron_config_rejects_mismatched_effective_global_batch() -> None:
    """Gradient accumulation should be derived from configured DP, not EP-expanded DP."""

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    config.training.gradient_accumulation_steps = 16

    import pytest

    with pytest.raises(ValueError, match="equal megatron.global_batch_size"):
        validate_megatron_config(config)


def test_megatron_config_rejects_unimplemented_model_parallel_degrees() -> None:
    """The manual loop should reject TP, PP, or CP degrees above one."""

    import pytest

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    config.megatron.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="requires megatron.tensor_model_parallel_size=1"):
        validate_megatron_config(config)


def test_megatron_sampler_uses_expert_data_parallel_domain(monkeypatch) -> None:
    """Sampler should shard data over configured DP replicas, excluding EP shards."""

    class FakeDist:
        """Fake distributed state for sampler construction."""

        @staticmethod
        def is_available() -> bool:
            """Return that distributed is available."""

            return True

        @staticmethod
        def is_initialized() -> bool:
            """Return that distributed is initialized."""

            return True

    import alf.megatron_train as megatron_train
    from megatron.core import parallel_state

    monkeypatch.setattr(megatron_train, "dist", FakeDist)
    monkeypatch.setattr(parallel_state, "get_expert_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_expert_data_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_world_size",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ordinary DP should not be used")),
    )

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    sampler = _build_megatron_sampler(list(range(8)), config)

    assert sampler.num_replicas == 2
    assert sampler.rank == 1


def test_megatron_ddp_optimizer_uses_torch_dtype_and_megatron_scheduler(monkeypatch) -> None:
    """Megatron-DDP optimizer branch should use torch_dtype and avoid LambdaLR."""

    import megatron.core.optimizer as megatron_optimizer

    captured = {}

    class FakeMegatronDdp:
        """Minimal object exposing Megatron DDP marker methods."""

        def zero_grad_buffer(self) -> None:
            """Fake buffer reset."""

        def finish_grad_sync(self) -> None:
            """Fake gradient sync."""

        def no_sync(self) -> None:
            """Fake no-sync context factory."""

    class FakeMegatronOptimizer:
        """Minimal Megatron optimizer facade for scheduler construction."""

        def __init__(self) -> None:
            """Create a torch-like param group list."""

            self.param_groups = [{"lr": 0.0}]

        def get_loss_scale(self) -> torch.Tensor:
            """Expose the Megatron optimizer marker used by the training loop."""

            return torch.tensor([1.0])

    def fake_get_megatron_optimizer(config, model_chunks, use_gloo_process_groups=True):
        """Capture Megatron optimizer arguments and return a fake optimizer."""

        captured["config"] = config
        captured["model_chunks"] = model_chunks
        captured["use_gloo_process_groups"] = use_gloo_process_groups
        return FakeMegatronOptimizer()

    monkeypatch.setattr(megatron_optimizer, "get_megatron_optimizer", fake_get_megatron_optimizer)
    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    config.model.torch_dtype = "bfloat16"
    config.training.learning_rate = 0.01
    config.training.warmup_steps = 2
    config.training.scheduler_type = "constant"
    config.training.max_grad_norm = None

    model = FakeMegatronDdp()
    optimizer = _build_megatron_training_optimizer(model, config)
    scheduler = _build_megatron_training_scheduler(optimizer, config)

    assert captured["config"].bf16 is True
    assert captured["config"].fp16 is False
    assert captured["config"].params_dtype is torch.bfloat16
    assert captured["config"].use_distributed_optimizer is config.megatron.distributed_optimizer
    assert captured["config"].clip_grad == 0.0
    assert captured["model_chunks"] == [model]
    assert captured["use_gloo_process_groups"] is False
    assert scheduler.__class__.__name__ == "MegatronLearningRateScheduler"
    assert optimizer.param_groups[0]["lr"] == 0.005
    scheduler.step()
    assert scheduler.get_last_lr() == [0.01]
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_megatron_clip_grad_normalizes_disabled_values() -> None:
    """Megatron clip config should preserve DDP disabled-clipping semantics."""

    assert _megatron_clip_grad(None) == 0.0
    assert _megatron_clip_grad(0.0) == 0.0
    assert _megatron_clip_grad(-1.0) == 0.0
    assert _megatron_clip_grad(1.5) == 1.5


def test_megatron_optimizer_skip_reports_no_actual_step() -> None:
    """Megatron optimizer overflow/skip should be visible to post-step hooks."""

    class FakeSkippedMegatronOptimizer:
        """Fake Megatron optimizer returning a skipped step result."""

        def get_loss_scale(self) -> torch.Tensor:
            """Expose the Megatron optimizer marker used by the training loop."""

            return torch.tensor([1.0])

        def step(self) -> tuple[bool, float, int]:
            """Return Megatron's skipped-step tuple."""

            return False, 7.5, 0

    step_successful, grad_norm = _step_megatron_optimizer(
        FakeSkippedMegatronOptimizer(),
        torch.nn.Linear(1, 1),
        max_grad_norm=1.0,
    )

    assert step_successful is False
    assert grad_norm == 7.5


def test_megatron_post_step_hooks_skip_when_optimizer_skips(monkeypatch) -> None:
    """ALF bias and scheduler should advance only after actual optimizer steps."""

    class FakeScheduler:
        """Fake scheduler that records step calls."""

        def __init__(self) -> None:
            """Initialize call counter."""

            self.steps = 0

        def step(self) -> None:
            """Record a scheduler step."""

            self.steps += 1

    import alf.megatron_train as megatron_train

    def fail_update(*args, **kwargs):
        """Fail if ALF bias update is called on a skipped optimizer step."""

        raise AssertionError("bias update should not run after skipped optimizer step")

    monkeypatch.setattr(megatron_train, "update_megatron_alf_router_biases", fail_update)
    scheduler = FakeScheduler()

    events = _post_megatron_optimizer_step(
        False,
        torch.nn.Linear(1, 1),
        scheduler,
        alf_enabled=True,
    )

    assert events == 0
    assert scheduler.steps == 0


def test_megatron_global_tokens_excludes_expert_parallel_duplicates() -> None:
    """Megatron throughput tokens should scale by configured DP, not EP*DP."""

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")

    assert _megatron_global_tokens(14, config) == 28



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


def test_megatron_core_alf_router_honors_max_update_step() -> None:
    """Megatron ALF bias should freeze after the configured absolute step."""

    from megatron.core.transformer.transformer_config import TransformerConfig

    config = ExperimentConfig(
        name="tiny-megatron-max-step",
        model=ModelConfig(
            hidden_size=2, intermediate_size=4, num_hidden_layers=1,
            num_attention_heads=1, num_key_value_heads=1, num_experts=4,
            num_experts_per_tok=3,
        ),
        megatron=MegatronConfig(
            enabled=True, expert_model_parallel_size=4, data_parallel_size=2,
            transformer_impl="local", moe_grouped_gemm=False,
            overlap_grad_reduce=False, overlap_param_gather=False,
        ),
        alf=AlfConfig(
            enabled=True, bias_update_rate=1.0, bias_update_policy="sign",
            bias_max_update_steps=1,
        ),
    )
    kwargs = megatron_transformer_config_kwargs(config)
    kwargs["perform_initialization"] = False
    router = MegatronCoreAuxiliaryLossFreeTopKRouter(
        config=TransformerConfig(**kwargs),
        pg_collection=SimpleNamespace(tp=None, cp=None, tp_cp=None, tp_dp_cp=None),
        alf_config=config.alf,
    )
    router.accumulated_expert_load.copy_(torch.tensor([3, 0, 0, 0]))
    assert router.update_expert_bias_from_accumulated_load() is True
    bias_after_first = router.expert_bias.clone()
    router.accumulated_expert_load.copy_(torch.tensor([0, 3, 0, 0]))
    assert router.update_expert_bias_from_accumulated_load() is False

    assert router.training_steps.item() == 2
    assert torch.equal(router.expert_bias, bias_after_first)


def test_megatron_core_alf_router_reduces_load_over_expert_dp(monkeypatch) -> None:
    """Core router should reduce accumulated load once over expert-DP."""

    from megatron.core.transformer.transformer_config import TransformerConfig

    class FakeReduceOp:
        """Fake torch distributed reduce operation namespace."""

        SUM = "sum"

    class FakeDist:
        """Fake distributed module validating the selected reduce group."""

        calls = 0

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
            """Mark expert-DP reductions by adding an offset."""

            assert op == "sum"
            assert group == "expert_dp"
            FakeDist.calls += 1
            tensor.add_(10)

    import alf.megatron_router as megatron_router

    monkeypatch.setattr(megatron_router, "dist", FakeDist)
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
        alf=AlfConfig(enabled=True, bias_update_rate=0.0),
    )
    kwargs = megatron_transformer_config_kwargs(config)
    kwargs["perform_initialization"] = False
    router = MegatronCoreAuxiliaryLossFreeTopKRouter(
        config=TransformerConfig(**kwargs),
        pg_collection=SimpleNamespace(tp=None, cp=None, tp_cp="ordinary", tp_dp_cp="ordinary", expt_dp="expert_dp"),
        alf_config=config.alf,
    )

    router.train()
    router.routing(torch.tensor([[[4.0, 3.0, 2.0, 1.0]]], device=router.weight.device))

    assert FakeDist.calls == 0
    assert router.accumulated_expert_load.cpu().tolist() == [1, 1, 1, 0]
    assert router.update_expert_bias_from_accumulated_load() is False
    assert FakeDist.calls == 1
    assert router.last_expert_load.cpu().tolist() == [11, 11, 11, 10]


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


def test_megatron_aux_loss_scale_matches_gradient_accumulation() -> None:
    """Injected auxiliary gradients should share main-loss GA scaling."""

    class FakeOptimizer:
        """Expose a deterministic mixed-precision loss scale."""

        def get_loss_scale(self) -> torch.Tensor:
            """Return the configured optimizer loss scale."""

            return torch.tensor(8.0)

    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    scale = _megatron_backward_scale(FakeOptimizer(), 4, torch.device("cpu"))
    _set_megatron_aux_loss_scale(scale)
    activation = torch.ones((), requires_grad=True)
    parameter = torch.ones((), requires_grad=True)
    output = MoEAuxLossAutoScaler.apply(activation, parameter)
    output.backward()

    assert scale.item() == 2.0
    assert parameter.grad.item() == 2.0


def test_megatron_checkpoint_roundtrip_restores_training_and_router_state(tmp_path) -> None:
    """Rank checkpoints should restore model, optimizer, scheduler, and step."""

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    model = torch.nn.Linear(2, 2)
    model.register_buffer("expert_bias", torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected_weight = model.weight.detach().clone()

    _save_megatron_rank_checkpoint(
        tmp_path, model, optimizer, scheduler, 7, config, attempt=9
    )
    with torch.no_grad():
        model.weight.zero_()
        model.expert_bias.zero_()

    step, attempt = _load_megatron_rank_checkpoint(
        tmp_path / "latest", model, optimizer, scheduler, config, torch.device("cpu")
    )

    assert (step, attempt) == (7, 9)
    assert torch.allclose(model.weight, expected_weight)
    assert model.expert_bias.tolist() == [1.0, -1.0]
    assert scheduler.last_epoch == 1

    _save_megatron_rank_checkpoint(
        tmp_path, model, optimizer, scheduler, 8, config, attempt=10
    )
    import json
    previous_metadata = json.loads(
        (tmp_path / ".latest.previous" / "metadata.json").read_text(encoding="utf-8")
    )
    assert previous_metadata["step"] == 7
    assert not (tmp_path / ".latest.incomplete").exists()


def test_megatron_checkpoint_rejects_incomplete_metadata(tmp_path) -> None:
    """Resume should fail before reading shards when publication is incomplete."""

    import json
    import pytest
    from alf.megatron_train import _checkpoint_topology

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")
    checkpoint = tmp_path / "latest"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "complete": False,
                "step": 3,
                "world_size": 1,
                "topology": _checkpoint_topology(config),
            }
        ),
        encoding="utf-8",
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with pytest.raises(RuntimeError, match="incomplete"):
        _load_megatron_rank_checkpoint(
            checkpoint, model, optimizer, scheduler, config, torch.device("cpu")
        )


def test_megatron_resume_defaults_to_latest_checkpoint(tmp_path) -> None:
    """Megatron should resume output-dir latest unless an explicit path wins."""

    config = load_experiment_config("experiments/qwen3_moe_c4_1b_megatron_alf.py")

    assert _resolve_megatron_resume_checkpoint(config, tmp_path) is None

    latest = tmp_path / "latest"
    latest.mkdir()
    assert _resolve_megatron_resume_checkpoint(config, tmp_path) == latest

    explicit = tmp_path / "selected-checkpoint"
    config.training.resume_from = str(explicit)
    assert _resolve_megatron_resume_checkpoint(config, tmp_path) == explicit


def test_megatron_load_observer_accumulates_across_microbatches() -> None:
    """Native-router observations should expose whole-step assignment counts."""

    class FakeTopKRouter(torch.nn.Module):
        """Small router exposing the Megatron router output contract."""

        def __init__(self) -> None:
            """Initialize one parameter and expert-count config."""

            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(num_moe_experts=3)

        def forward(self, routing_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return probabilities and the supplied boolean routing map."""

            return routing_map.float(), routing_map

    router = FakeTopKRouter()
    _install_megatron_load_observers(router)
    from alf.megatron_train import _set_megatron_load_observation
    _set_megatron_load_observation(router, True)
    router(torch.tensor([[True, False, True]]))
    router(torch.tensor([[False, True, True]]))

    counts = _collect_megatron_load_observers(router)

    assert counts[""].tolist() == [1, 1, 2]


def test_megatron_alf_updates_stack_layer_reduction(monkeypatch) -> None:
    """ALF should reduce all same-group layer counts in one collective."""

    import alf.megatron_router as megatron_router

    calls = []

    class FakeRouter:
        """Router facade recording a pre-reduced ALF update."""

        def __init__(self, counts: list[int]) -> None:
            """Initialize local counts in a shared fake group."""

            self.accumulated_expert_load = torch.tensor(counts)
            self.alf_load_group = "expert_dp"
            self.reduced = None

        def update_expert_bias_from_reduced_load(self, counts: torch.Tensor) -> bool:
            """Store the reduced counts and report one update event."""

            self.reduced = counts.clone()
            return True

    routers = [FakeRouter([1, 2]), FakeRouter([3, 4])]

    def fake_reduce(counts: torch.Tensor, group: object) -> torch.Tensor:
        """Record one stacked reduction and return a deterministic result."""

        calls.append((counts.clone(), group))
        return counts + 10

    monkeypatch.setattr(
        megatron_router,
        "iter_megatron_alf_routers",
        lambda module: iter([("a", routers[0]), ("b", routers[1])]),
    )
    monkeypatch.setattr(megatron_router, "reduce_expert_load_counts", fake_reduce)

    events = update_megatron_alf_router_biases(torch.nn.Linear(1, 1))

    assert events == 2
    assert len(calls) == 1
    assert calls[0][0].shape == (2, 2)
    assert routers[0].reduced.tolist() == [11, 12]
    assert routers[1].reduced.tolist() == [13, 14]
