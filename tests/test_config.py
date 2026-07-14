"""Tests for Python experiment configuration loading."""

from alf.config import ExperimentConfig, load_experiment_config


def test_load_experiment_config() -> None:
    """Load the default ALF experiment file."""

    config = load_experiment_config("experiments/qwen3_moe_tiny_alf.py")

    assert isinstance(config, ExperimentConfig)
    assert config.name == "qwen3_moe_tiny_alf"
    assert config.alf.enabled is True


def test_apply_dotted_overrides() -> None:
    """Apply safe dotted CLI overrides to typed fields."""

    config = load_experiment_config(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "2",
            "--alf.enabled=false",
            "--model.router_aux_loss_coef",
            "0.02",
            "--alf.bias_update_schedule",
            "linear",
            "--alf.bias_update_schedule_steps",
            "10",
            "--alf.bias_update_end_rate=1e-5",
            "--alf.bias_max_update_steps",
            "7",
            "--alf.bias_adaptive_beta_min",
            "0.2",
            "--alf.bias_adaptive_beta_max",
            "0.8",
            "--alf.bias_adaptive_variance_reference",
            "0.01",
            "--alf.bias_adaptive_state_decay",
            "0.7",
            "--alf.bias_adaptive_per_expert_beta",
            "0.6",
            "--alf.bias_adaptive_per_expert_momentum_beta",
            "0.7",
            "--alf.bias_adaptive_per_expert_epsilon",
            "1e-6",
            "--data.train_files=tests/fixtures/tiny_corpus.txt",
            "--training.num_workers",
            "2",
            "--training.pin_memory",
            "true",
            "--training.gradient_checkpointing=true",
            "--training.max_grad_norm",
            "0.5",
            "--training.optimizer_state_dtype",
            "parameter",
            "--training.ddp_backend",
            "gloo",
            "--megatron.enabled",
            "true",
            "--megatron.expert_model_parallel_size",
            "4",
            "--megatron.data_parallel_size",
            "2",
            "--megatron.global_batch_size",
            "16",
        ],
    )

    assert config.training.max_steps == 2
    assert config.alf.enabled is False
    assert config.model.router_aux_loss_coef == 0.02
    assert config.alf.bias_update_schedule == "linear"
    assert config.alf.bias_update_schedule_steps == 10
    assert config.alf.bias_update_end_rate == 1e-5
    assert config.alf.bias_max_update_steps == 7
    assert config.alf.bias_adaptive_beta_min == 0.2
    assert config.alf.bias_adaptive_beta_max == 0.8
    assert config.alf.bias_adaptive_variance_reference == 0.01
    assert config.alf.bias_adaptive_state_decay == 0.7
    assert config.alf.bias_adaptive_per_expert_beta == 0.6
    assert config.alf.bias_adaptive_per_expert_momentum_beta == 0.7
    assert config.alf.bias_adaptive_per_expert_epsilon == 1e-6
    assert config.data.train_files == ["tests/fixtures/tiny_corpus.txt"]
    assert config.training.num_workers == 2
    assert config.training.pin_memory is True
    assert config.training.gradient_checkpointing is True
    assert config.training.max_grad_norm == 0.5
    assert config.training.optimizer_state_dtype == "parameter"
    assert config.training.ddp_backend == "gloo"
    assert config.megatron.enabled is True
    assert config.megatron.expert_model_parallel_size == 4
    assert config.megatron.data_parallel_size == 2
    assert config.megatron.global_batch_size == 16
