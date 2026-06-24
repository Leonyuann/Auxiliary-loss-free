"""Tiny Qwen3 MoE traditional auxiliary-loss baseline experiment."""

from alf.config import AlfConfig, DataConfig, ExperimentConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    name="qwen3_moe_tiny_aux_loss",
    model=ModelConfig(use_tiny_config=True),
    data=DataConfig(
        train_files=["tests/fixtures/tiny_corpus.txt"],
        block_size=32,
        max_train_samples=8,
    ),
    training=TrainingConfig(
        output_dir="outputs/qwen3_moe_tiny_aux_loss",
        max_steps=5,
        batch_size=2,
        learning_rate=3e-4,
        log_every=1,
        save_every=5,
    ),
    alf=AlfConfig(
        enabled=False,
        disable_router_aux_loss=False,
    ),
)
