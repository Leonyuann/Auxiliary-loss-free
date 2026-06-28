"""Tiny Qwen3 MoE auxiliary-loss-free training experiment."""

from alf.config import AlfConfig, DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainingConfig, WandbConfig

config = ExperimentConfig(
    name="qwen3_moe_tiny_alf",
    model=ModelConfig(use_tiny_config=True),
    data=DataConfig(
        train_files=["tests/fixtures/tiny_corpus.txt"],
        validation_files=["tests/fixtures/tiny_corpus.txt"],
        block_size=32,
        max_train_samples=8,
        max_validation_samples=4,
    ),
    eval=EvalConfig(eval_every=5, eval_batch_size=2),
    training=TrainingConfig(
        output_dir="outputs/qwen3_moe_tiny_alf",
        max_steps=5,
        batch_size=2,
        learning_rate=3e-4,
        log_every=1,
        save_every=5,
    ),
    alf=AlfConfig(
        enabled=True,
        bias_update_rate=1e-3,
        bias_update_policy="proportional",
        bias_clip=1.0,
        disable_router_aux_loss=True,
    ),
    wandb=WandbConfig(enabled=True, tags=["alf", "qwen3-moe", "tiny"]),
)
