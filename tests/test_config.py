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
            "--alf.bias_update_schedule",
            "linear",
            "--alf.bias_update_schedule_steps",
            "10",
            "--alf.bias_update_end_rate=1e-5",
            "--data.train_files=tests/fixtures/tiny_corpus.txt",
            "--training.num_workers",
            "2",
            "--training.pin_memory",
            "true",
            "--training.gradient_checkpointing=true",
            "--training.max_grad_norm",
            "0.5",
            "--training.ddp_backend",
            "gloo",
        ],
    )

    assert config.training.max_steps == 2
    assert config.alf.enabled is False
    assert config.alf.bias_update_schedule == "linear"
    assert config.alf.bias_update_schedule_steps == 10
    assert config.alf.bias_update_end_rate == 1e-5
    assert config.data.train_files == ["tests/fixtures/tiny_corpus.txt"]
    assert config.training.num_workers == 2
    assert config.training.pin_memory is True
    assert config.training.gradient_checkpointing is True
    assert config.training.max_grad_norm == 0.5
    assert config.training.ddp_backend == "gloo"
