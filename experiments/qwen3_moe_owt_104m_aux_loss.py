"""104M-parameter Qwen3 MoE OWT traditional auxiliary-loss 4-hour experiment."""

from alf.config import AlfConfig, DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainingConfig, WandbConfig

config = ExperimentConfig(
    name="qwen3_moe_owt_104m_aux_loss",
    model=ModelConfig(
        use_tiny_config=True,
        tokenizer_name_or_path="/vepfs-mlp2/ylq/tokenizers/owt_bpe_32k",
        vocab_size=32768,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        torch_dtype="bfloat16",
    ),
    data=DataConfig(
        train_files=["/vepfs-mlp2/ylq/data/train_1310m_bpe32k_tokens.i32"],
        validation_files=["/vepfs-mlp2/ylq/data/validation_16m_bpe32k_tokens.i32"],
        block_size=512,
        max_train_samples=2_560_000,
        max_validation_samples=32_768,
    ),
    eval=EvalConfig(eval_every=500, eval_batch_size=16, max_eval_samples=2048),
    training=TrainingConfig(
        output_dir="outputs/qwen3_moe_owt_104m_aux_loss",
        seed=42,
        max_steps=10_000,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        weight_decay=0.1,
        scheduler_type="cosine",
        warmup_steps=500,
        log_every=10,
        save_every=1000,
        device="auto",
    ),
    alf=AlfConfig(
        enabled=False,
        disable_router_aux_loss=False,
    ),
    wandb=WandbConfig(
        enabled=True,
        entity="liangqingyuann-huazhong-university-of-science-and-technology",
        project="Load-balance",
        tags=["aux-loss", "qwen3-moe", "owt", "104m", "4h", "bpe32k"],
    ),
)
