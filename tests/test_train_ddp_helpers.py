"""Tests for DDP training helper behavior."""

from __future__ import annotations

import pytest

from alf.config import AlfConfig, ExperimentConfig, TrainingConfig, load_experiment_config
from alf.train import DistributedState, _build_train_sampler, _is_main_process, _unwrap_model, _validate_training_config

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


def test_c4_alf_configs_disable_gradient_checkpointing() -> None:
    """C4 ALF configs should preserve one bias update per training forward."""

    for path in [
        "experiments/qwen3_moe_c4_500m_alf.py",
        "experiments/qwen3_moe_c4_500m_alf_ema.py",
    ]:
        config = load_experiment_config(path)
        assert config.alf.enabled is True
        assert config.training.gradient_checkpointing is False
