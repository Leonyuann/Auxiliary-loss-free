"""Typed Python experiment configuration loading and CLI overrides."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints


@dataclass
class ModelConfig:
    """Model construction options.

    Attributes:
        model_name_or_path: Optional Hugging Face model id or local checkpoint path.
        tokenizer_name_or_path: Optional tokenizer id or local tokenizer path.
        use_tiny_config: Whether to initialize a tiny local Qwen3 MoE config.
        vocab_size: Tiny model vocabulary size.
        hidden_size: Tiny model hidden size.
        intermediate_size: Tiny model expert intermediate size.
        num_hidden_layers: Number of decoder layers for tiny config.
        num_attention_heads: Number of attention heads for tiny config.
        num_key_value_heads: Number of key/value attention heads for tiny config.
        num_experts: Number of routed experts for tiny config.
        num_experts_per_tok: Number of experts selected per token.
        torch_dtype: Training dtype name, such as ``float32`` or ``bfloat16``.
        trust_remote_code: Whether Hugging Face loading may use remote code.
    """

    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    use_tiny_config: bool = True
    vocab_size: int = 1024
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    num_experts: int = 4
    num_experts_per_tok: int = 2
    torch_dtype: str = "float32"
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    """Data loading and token packing options.

    Attributes:
        train_files: Text files used for causal language model training.
        validation_files: Text files used for validation metrics.
        block_size: Fixed token block size after packing.
        max_train_samples: Optional maximum number of packed blocks to keep.
        max_validation_samples: Optional maximum number of validation blocks.
        tokenizer_name_or_path: Optional tokenizer override.
    """

    train_files: list[str] = field(default_factory=lambda: ["tests/fixtures/tiny_corpus.txt"])
    validation_files: list[str] = field(default_factory=lambda: ["tests/fixtures/tiny_corpus.txt"])
    block_size: int = 64
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    tokenizer_name_or_path: str | None = None


@dataclass
class TrainingConfig:
    """Training loop options.

    Attributes:
        output_dir: Directory for checkpoints and metrics.
        seed: Random seed.
        max_steps: Number of optimizer steps.
        batch_size: Per-step batch size.
        gradient_accumulation_steps: Number of backward passes per optimizer step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        log_every: Step interval for console and metrics logging.
        save_every: Step interval for checkpoint saving.
        resume_from: Optional checkpoint directory to resume from.
        device: Device string. Use ``auto`` to prefer CUDA when available.
        num_workers: Number of dataloader worker processes.
        pin_memory: Whether dataloaders should pin CPU memory.
        drop_last: Whether dataloaders and distributed samplers should drop the
            final incomplete batch.
        gradient_checkpointing: Whether to enable model gradient checkpointing.
        ddp_backend: Optional torch.distributed backend override.
        ddp_find_unused_parameters: Whether DDP should search for unused
            parameters during backward.
    """

    output_dir: str = "outputs/qwen3_moe_tiny_alf"
    seed: int = 42
    max_steps: int = 5
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    scheduler_type: str = "constant"
    log_every: int = 1
    save_every: int = 5
    resume_from: str | None = None
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    gradient_checkpointing: bool = False
    ddp_backend: str | None = None
    ddp_find_unused_parameters: bool = False


@dataclass
class AlfConfig:
    """Auxiliary-loss-free router options.

    Attributes:
        enabled: Whether to replace routers with auxiliary-loss-free routers.
        bias_update_rate: Magnitude of each load-balancing bias update.
        bias_update_policy: Bias update policy name. Supported values are
            ``proportional``, ``sign``, ``ema``, ``accumulated_sign``, and
            ``balanced_topk_sign``.
        bias_ema_beta: EMA coefficient used by the ``ema`` policy.
        bias_update_topk: Number of positive-error and negative-error experts
            updated by the ``balanced_topk_sign`` policy.
        bias_update_schedule: Schedule for the bias update rate. Supported values are
            ``constant`` and ``linear``.
        bias_update_schedule_steps: Number of post-warmup router calls used by the
            bias update schedule. Required for ``linear``.
        bias_update_end_rate: Final bias update rate for scheduled decay.
        bias_init: Initial expert bias value.
        bias_clip: Optional absolute clipping limit for expert bias.
        update_interval: Number of router calls between bias updates.
        warmup_steps: Number of router calls before bias updates begin.
        disable_router_aux_loss: Whether to set the original router aux loss to zero.
    """

    enabled: bool = True
    bias_update_rate: float = 1e-3
    bias_update_policy: str = "proportional"
    bias_ema_beta: float = 0.9
    bias_update_topk: int = 1
    bias_update_schedule: str = "constant"
    bias_update_schedule_steps: int | None = None
    bias_update_end_rate: float = 0.0
    bias_init: float = 0.0
    bias_clip: float | None = None
    update_interval: int = 1
    warmup_steps: int = 0
    disable_router_aux_loss: bool = True


@dataclass
class EvalConfig:
    """Validation evaluation options.

    Attributes:
        eval_every: Step interval for validation. Use zero to disable periodic eval.
        eval_batch_size: Validation dataloader batch size.
        max_eval_samples: Optional maximum validation examples to evaluate.
    """

    eval_every: int = 5
    eval_batch_size: int = 2
    max_eval_samples: int | None = None


@dataclass
class WandbConfig:
    """Weights & Biases logging options.

    Attributes:
        enabled: Whether W&B logging is enabled.
        entity: Optional W&B entity. Defaults to ``WANDB_ENTITY``.
        project: Optional W&B project. Defaults to ``WANDB_PROJECT``.
        mode: W&B mode, such as ``online``, ``offline``, or ``disabled``.
        group: Optional run group. Defaults to ``WANDB_RUN_GROUP``.
        tags: Optional run tags. Defaults to comma-separated ``WANDB_TAGS``.
        log_checkpoints: Whether checkpoint artifacts should be uploaded.
    """

    enabled: bool = True
    entity: str | None = "liangqingyuann-huazhong-university-of-science-and-technology"
    project: str | None = "Load-balance"
    mode: str = "online"
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    log_checkpoints: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Attributes:
        name: Human-readable experiment name.
        model: Model construction options.
        data: Data loading options.
        training: Training loop options.
        alf: Auxiliary-loss-free routing options.
        eval: Validation evaluation options.
        wandb: W&B logging options.
    """

    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alf: AlfConfig = field(default_factory=AlfConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def parse_config_args(argv: Sequence[str] | None = None) -> tuple[Path, list[str]]:
    """Parse a Python experiment path and dotted override arguments.

    Args:
        argv: Optional command-line argument sequence.

    Returns:
        The experiment file path and a list of raw dotted overrides.
    """

    parser = argparse.ArgumentParser(description="Run an ALF experiment.")
    parser.add_argument("experiment", type=Path, help="Python file exporting `config`.")
    args, overrides = parser.parse_known_args(argv)
    return args.experiment, overrides


def load_experiment_config(path: str | Path, overrides: Sequence[str] | None = None) -> ExperimentConfig:
    """Load an experiment config object from a Python file.

    Args:
        path: Python experiment file path.
        overrides: Optional dotted CLI overrides such as ``--training.max_steps 2``.

    Returns:
        The loaded and overridden experiment config.

    Raises:
        FileNotFoundError: If the experiment file does not exist.
        AttributeError: If the file does not export ``config``.
        TypeError: If ``config`` is not an ``ExperimentConfig``.
        ValueError: If an override is malformed or targets an unknown field.
    """

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {config_path}")

    module = _load_module_from_path(config_path)
    if not hasattr(module, "config"):
        raise AttributeError(f"Experiment file must export `config`: {config_path}")

    config = module.config
    if not isinstance(config, ExperimentConfig):
        raise TypeError(f"`config` must be ExperimentConfig, got {type(config)!r}")

    if overrides:
        apply_overrides(config, overrides)
    return config


def apply_overrides(config: ExperimentConfig, overrides: Sequence[str]) -> None:
    """Apply dotted command-line overrides to a dataclass config in place.

    Args:
        config: Experiment config to mutate.
        overrides: Raw overrides in ``--section.field value`` or
            ``--section.field=value`` form.

    Raises:
        ValueError: If an override is malformed.
        AttributeError: If an override targets an unknown field.
    """

    index = 0
    while index < len(overrides):
        key = overrides[index]
        if not key.startswith("--"):
            raise ValueError(f"Override keys must start with `--`: {key}")
        key = key[2:]
        if "=" in key:
            dotted_key, raw_value = key.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(overrides):
                raise ValueError(f"Override missing value: --{key}")
            dotted_key = key
            raw_value = overrides[index + 1]
            index += 2
        _set_dotted_value(config, dotted_key, raw_value)


def asdict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert an experiment config to a plain dictionary.

    Args:
        config: Experiment config.

    Returns:
        Dataclass content as a nested dictionary.
    """

    return dataclasses.asdict(config)


def _load_module_from_path(path: Path) -> ModuleType:
    """Import a Python module from a file path.

    Args:
        path: Python file path.

    Returns:
        Imported module.
    """

    module_name = f"alf_experiment_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _set_dotted_value(config: ExperimentConfig, dotted_key: str, raw_value: str) -> None:
    """Set one dotted dataclass field.

    Args:
        config: Root experiment config.
        dotted_key: Dotted field path.
        raw_value: String value from CLI.

    Raises:
        AttributeError: If any field is unknown.
    """

    parts = dotted_key.split(".")
    if len(parts) < 1:
        raise ValueError(f"Invalid override key: {dotted_key}")

    target: Any = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(f"Unknown config section in override: {dotted_key}")
        target = getattr(target, part)

    field_name = parts[-1]
    if not hasattr(target, field_name):
        raise AttributeError(f"Unknown config field in override: {dotted_key}")

    field_type = _field_type(target, field_name)
    setattr(target, field_name, _coerce_value(raw_value, field_type))


def _field_type(target: Any, field_name: str) -> Any:
    """Return the declared dataclass field type.

    Args:
        target: Dataclass instance.
        field_name: Field name.

    Returns:
        Declared field type or ``str`` when unavailable.
    """

    hints = get_type_hints(type(target))
    return hints.get(field_name, str)


def _coerce_value(raw_value: str, field_type: Any) -> Any:
    """Coerce a CLI string value to the target field type.

    Args:
        raw_value: Raw CLI value.
        field_type: Dataclass field type.

    Returns:
        Coerced Python value.
    """

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin in {list, Sequence}:
        item_type = args[0] if args else str
        return [_coerce_value(part.strip(), item_type) for part in raw_value.split(",") if part.strip()]

    if origin in {UnionType, getattr(__import__("typing"), "Union")} or (
        origin is None and hasattr(field_type, "__args__")
    ):
        args = tuple(arg for arg in get_args(field_type) if arg is not type(None))
        if len(args) == 1:
            return None if raw_value.lower() == "none" else _coerce_value(raw_value, args[0])

    if str(field_type).startswith("typing.Optional") and raw_value.lower() == "none":
        return None

    if field_type is bool:
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if field_type is int:
        return int(raw_value)
    if field_type is float:
        return float(raw_value)
    if field_type is str:
        return raw_value
    if raw_value.lower() == "none":
        return None
    return raw_value
