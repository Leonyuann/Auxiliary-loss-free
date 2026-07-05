"""Tests for DDP training helper behavior."""

from __future__ import annotations

import pytest

from alf.config import AlfConfig, ExperimentConfig, TrainingConfig, load_experiment_config
from alf.train import (
    FP32AdamW,
    DistributedState,
    _build_optimizer,
    _build_train_sampler,
    _clip_or_measure_gradient_norm,
    _gradient_norm,
    _is_main_process,
    _unwrap_model,
    _validate_training_config,
)

import torch


def test_build_train_sampler_uses_distributed_state() -> None:
    """Distributed sampler should use rank, world size, seed, and drop-last config."""

    config = ExperimentConfig(
        name="ddp-test",
        training=TrainingConfig(seed=123, drop_last=True),
    )
    state = DistributedState(enabled=True, rank=1, local_rank=1, world_size=2, is_main=False)

    sampler = _build_train_sampler(list(range(8)), config, state)

    assert sampler is not None
    assert sampler.rank == 1
    assert sampler.num_replicas == 2
    assert sampler.seed == 123
    assert sampler.drop_last is True
    assert _is_main_process(state) is False


def test_single_process_sampler_and_unwrap_are_noops() -> None:
    """Single-process helper behavior should preserve existing training defaults."""

    config = ExperimentConfig(name="single-test")
    model = torch.nn.Linear(2, 2)

    assert _build_train_sampler(list(range(4)), config, DistributedState()) is None
    assert _is_main_process(DistributedState()) is True
    assert _unwrap_model(model) is model


def test_alf_rejects_gradient_checkpointing_side_effects() -> None:
    """ALF bias side effects should not run inside checkpointed forwards."""

    config = ExperimentConfig(
        name="alf-checkpointing",
        alf=AlfConfig(enabled=True),
        training=TrainingConfig(gradient_checkpointing=True),
    )

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        _validate_training_config(config)


def test_c4_configs_disable_gradient_checkpointing_for_fair_comparison() -> None:
    """C4 baselines should use the same checkpointing mode for throughput comparisons."""

    for path in [
        "experiments/qwen3_moe_c4_300m_alf.py",
        "experiments/qwen3_moe_c4_300m_alf_ema.py",
        "experiments/qwen3_moe_c4_300m_aux_loss.py",
    ]:
        config = load_experiment_config(path)
        assert config.training.gradient_checkpointing is False


def test_c4_300m_configs_use_reasonable_moe_scale() -> None:
    """C4 300M-family configs should use a 16-expert MoE and shorter C4 run."""

    for path in [
        "experiments/qwen3_moe_c4_300m_alf.py",
        "experiments/qwen3_moe_c4_300m_alf_ema.py",
        "experiments/qwen3_moe_c4_300m_aux_loss.py",
    ]:
        config = load_experiment_config(path)
        assert config.model.hidden_size == 512
        assert config.model.intermediate_size == 1280
        assert config.model.num_hidden_layers == 9
        assert config.model.num_attention_heads == 8
        assert config.model.num_key_value_heads == 4
        assert config.model.num_experts == 16
        assert config.model.num_experts_per_tok == 2
        assert config.training.max_steps == 100_000
        assert config.training.warmup_steps == 2000
        assert config.training.max_grad_norm == 1.0
        assert config.training.optimizer_state_dtype == "float32"
        assert config.training.save_every == 2500
        assert config.eval.eval_every == 1000
        assert config.eval.eval_batch_size == 32


def test_c4_alf_bias_update_cadence_is_stable_for_accumulation() -> None:
    """C4 ALF configs should avoid overly frequent bias updates under accumulation."""

    sign_config = load_experiment_config("experiments/qwen3_moe_c4_300m_alf.py")
    ema_config = load_experiment_config("experiments/qwen3_moe_c4_300m_alf_ema.py")

    assert sign_config.alf.bias_update_policy == "sign"
    assert sign_config.alf.bias_update_rate == 5e-4
    assert sign_config.alf.bias_update_end_rate == 1e-4
    assert sign_config.alf.bias_update_schedule == "linear"
    assert sign_config.alf.bias_update_schedule_steps == 200_000
    assert sign_config.alf.update_interval == sign_config.training.gradient_accumulation_steps
    assert sign_config.alf.warmup_steps == 4000
    assert sign_config.alf.bias_clip == 2.0

    assert ema_config.alf.bias_update_policy == "ema"
    assert ema_config.alf.bias_update_rate == 1e-2
    assert ema_config.alf.bias_update_end_rate == 1e-3
    assert ema_config.alf.bias_ema_beta == 0.9
    assert ema_config.alf.bias_update_schedule == "linear"
    assert ema_config.alf.bias_update_schedule_steps == 200_000
    assert ema_config.alf.update_interval == ema_config.training.gradient_accumulation_steps
    assert ema_config.alf.warmup_steps == 4000
    assert ema_config.alf.bias_clip == 2.0


def test_clip_or_measure_gradient_norm_clips_when_configured() -> None:
    """Gradient clipping should cap gradients and report the pre-clip norm."""

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    grad_norm = _clip_or_measure_gradient_norm(model, max_grad_norm=1.0)

    assert grad_norm == pytest.approx(5.0)
    assert _gradient_norm(model) == pytest.approx(1.0)


def test_clip_or_measure_gradient_norm_can_be_disabled() -> None:
    """Non-positive clipping thresholds should preserve current gradients."""

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    grad_norm = _clip_or_measure_gradient_norm(model, max_grad_norm=0.0)

    assert grad_norm == pytest.approx(5.0)
    assert _gradient_norm(model) == pytest.approx(5.0)


def test_build_optimizer_keeps_bfloat16_adamw_state_in_float32() -> None:
    """BF16 model parameters should use FP32 AdamW master and moment state."""

    model = torch.nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
    optimizer = _build_optimizer(
        model,
        learning_rate=0.1,
        weight_decay=0.0,
        optimizer_state_dtype="float32",
    )
    model.weight.grad = torch.ones_like(model.weight)

    optimizer.step()

    assert isinstance(optimizer, FP32AdamW)
    state = optimizer.state[model.weight]
    assert model.weight.dtype == torch.bfloat16
    assert state["master_param"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_build_optimizer_can_use_parameter_dtype_state() -> None:
    """The opt-out mode should preserve PyTorch's native AdamW behavior."""

    model = torch.nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
    optimizer = _build_optimizer(
        model,
        learning_rate=0.1,
        weight_decay=0.0,
        optimizer_state_dtype="parameter",
    )
    model.weight.grad = torch.ones_like(model.weight)

    optimizer.step()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert not isinstance(optimizer, FP32AdamW)
    assert optimizer.state[model.weight]["exp_avg"].dtype == torch.bfloat16


def test_fp32_adamw_loads_legacy_low_precision_state_as_float32() -> None:
    """Legacy native AdamW checkpoints should resume with FP32 optimizer state."""

    model = torch.nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
    native_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model.weight.grad = torch.ones_like(model.weight)
    native_optimizer.step()
    native_state = native_optimizer.state_dict()

    replacement = FP32AdamW(model.parameters(), lr=0.1)
    replacement.load_state_dict(native_state)

    state = replacement.state[model.weight]
    assert state["master_param"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32

