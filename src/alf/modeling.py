"""Model patching helpers for auxiliary-loss-free Qwen3 MoE routing."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

from .config import AlfConfig, ModelConfig
from .router import Qwen3MoeAuxiliaryLossFreeTopKRouter

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
except ImportError:  # pragma: no cover - exercised indirectly in environments with transformers.
    AutoModelForCausalLM = None
    AutoTokenizer = None
    Qwen3MoeConfig = None
    Qwen3MoeForCausalLM = None
    Qwen3MoeTopKRouter = None


class SimpleByteTokenizer:
    """Small deterministic tokenizer for local tiny-model smoke tests.

    Attributes:
        vocab_size: Number of token ids available.
        eos_token_id: End-of-sequence token id.
        pad_token_id: Padding token id.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the tokenizer.

        Args:
            vocab_size: Vocabulary size used by the tiny model.
        """

        self.vocab_size = int(vocab_size)
        self.eos_token_id = 1
        self.pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        """Encode text into deterministic byte-level ids.

        Args:
            text: Input text.
            add_special_tokens: Whether to append EOS during encoding.

        Returns:
            Tokenizer-style dictionary containing ``input_ids``.
        """

        usable_vocab = max(self.vocab_size - 2, 1)
        input_ids = [(byte % usable_vocab) + 2 for byte in text.encode("utf-8")]
        if add_special_tokens:
            input_ids.append(self.eos_token_id)
        return {"input_ids": input_ids}


def _iter_named_children_with_parents(
    module: nn.Module,
    prefix: str = "",
) -> Iterator[tuple[nn.Module, str, str, nn.Module]]:
    """Yield child modules together with their parent module and qualified name.

    Args:
        module: Root module to traverse.
        prefix: Existing module-name prefix.

    Yields:
        Tuples of `(parent_module, child_name, qualified_name, child_module)`.
    """

    for child_name, child_module in module.named_children():
        qualified_name = child_name if not prefix else f"{prefix}.{child_name}"
        yield module, child_name, qualified_name, child_module
        yield from _iter_named_children_with_parents(child_module, qualified_name)


def is_qwen3_moe_router(module: nn.Module) -> bool:
    """Return whether a module matches the Qwen3 MoE router contract.

    Args:
        module: Module to inspect.

    Returns:
        `True` when the module is a Qwen3 MoE top-k router or a structural match.
    """

    if Qwen3MoeTopKRouter is not None and isinstance(module, Qwen3MoeTopKRouter):
        return True

    required_attributes = ("weight", "top_k", "num_experts", "norm_topk_prob", "hidden_dim")
    if not all(hasattr(module, attribute) for attribute in required_attributes):
        return False

    weight = getattr(module, "weight")
    if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        return False

    return int(module.num_experts) == int(weight.shape[0]) and int(module.hidden_dim) == int(weight.shape[1])


def iter_auxiliary_loss_free_routers(
    module: nn.Module,
) -> Iterator[tuple[str, Qwen3MoeAuxiliaryLossFreeTopKRouter]]:
    """Yield every auxiliary-loss-free Qwen3 router in a module tree.

    Args:
        module: Root module to traverse.

    Yields:
        Tuples of `(qualified_name, router_module)`.
    """

    for qualified_name, child_module in module.named_modules():
        if isinstance(child_module, Qwen3MoeAuxiliaryLossFreeTopKRouter):
            yield qualified_name, child_module


def disable_router_aux_loss(model: nn.Module) -> bool:
    """Disable the Hugging Face router auxiliary-loss coefficient when present.

    Args:
        model: Model whose auxiliary-loss coefficient should be disabled.

    Returns:
        `True` when at least one attribute was updated.
    """

    disabled = False
    if hasattr(model, "router_aux_loss_coef"):
        setattr(model, "router_aux_loss_coef", 0.0)
        disabled = True
    if hasattr(model, "config") and hasattr(model.config, "router_aux_loss_coef"):
        setattr(model.config, "router_aux_loss_coef", 0.0)
        disabled = True
    return disabled


def replace_qwen3_moe_routers(
    model: nn.Module,
    *,
    alf_enabled: bool = True,
    expert_bias_init: float = 0.0,
    expert_bias_update_rate: float = 0.0,
    expert_bias_update_policy: str = "proportional",
    expert_bias_update_interval: int = 1,
    expert_bias_ema_beta: float = 0.9,
    expert_bias_update_topk: int = 1,
    expert_bias_update_schedule: str = "constant",
    expert_bias_update_schedule_steps: int | None = None,
    expert_bias_update_end_rate: float = 0.0,
    expert_bias_clip: float | None = None,
    expert_bias_warmup_steps: int = 0,
    disable_original_router_aux_loss: bool = True,
) -> dict[str, Any]:
    """Replace compatible Qwen3 MoE routers with auxiliary-loss-free routers.

    Args:
        model: Loaded model to patch in place.
        alf_enabled: Whether auxiliary-loss-free routing is enabled for the model.
        expert_bias_init: Initial scalar value copied into all expert bias entries.
        expert_bias_update_rate: Update magnitude used for load-balancing bias steps.
        expert_bias_update_policy: Bias update policy.
        expert_bias_update_interval: Number of training forwards between updates.
        expert_bias_ema_beta: EMA coefficient for the ``ema`` bias update policy.
        expert_bias_update_topk: Number of positive-error and negative-error experts
            updated by the ``balanced_topk_sign`` policy.
        expert_bias_update_schedule: Schedule used for bias update rates.
        expert_bias_update_schedule_steps: Number of post-warmup training forwards
            used by the schedule.
        expert_bias_update_end_rate: Final bias update rate for scheduled decay.
        expert_bias_clip: Optional symmetric clip magnitude for bias entries.
        expert_bias_warmup_steps: Number of training forwards to skip before updates.
        disable_original_router_aux_loss: Whether to zero the original router
            auxiliary-loss coefficient when ALF routing is enabled.

    Returns:
        A serializable summary describing which routers were replaced.
    """

    replaced_router_paths: list[str] = []
    router_aux_loss_disabled = (
        disable_router_aux_loss(model) if alf_enabled and disable_original_router_aux_loss else False
    )

    if not alf_enabled:
        return {
            "alf_enabled": False,
            "num_replaced": 0,
            "replaced_router_paths": replaced_router_paths,
            "router_aux_loss_disabled": router_aux_loss_disabled,
        }

    for parent_module, child_name, qualified_name, child_module in _iter_named_children_with_parents(model):
        if not is_qwen3_moe_router(child_module):
            continue
        replacement_router = Qwen3MoeAuxiliaryLossFreeTopKRouter.from_qwen3_router(
            child_module,
            expert_bias_init=expert_bias_init,
            expert_bias_update_rate=expert_bias_update_rate,
            expert_bias_update_policy=expert_bias_update_policy,
            expert_bias_update_interval=expert_bias_update_interval,
            expert_bias_ema_beta=expert_bias_ema_beta,
            expert_bias_update_topk=expert_bias_update_topk,
            expert_bias_update_schedule=expert_bias_update_schedule,
            expert_bias_update_schedule_steps=expert_bias_update_schedule_steps,
            expert_bias_update_end_rate=expert_bias_update_end_rate,
            expert_bias_clip=expert_bias_clip,
            expert_bias_warmup_steps=expert_bias_warmup_steps,
        )
        setattr(parent_module, child_name, replacement_router)
        replaced_router_paths.append(qualified_name)

    return {
        "alf_enabled": True,
        "num_replaced": len(replaced_router_paths),
        "replaced_router_paths": replaced_router_paths,
        "router_aux_loss_disabled": router_aux_loss_disabled,
    }


def _config_value(config: Any, name: str, default: Any) -> Any:
    """Read a setting from a mapping, dataclass, or plain object.

    Args:
        config: Config object to inspect.
        name: Setting name to read.
        default: Value returned when the setting is absent.

    Returns:
        The configured value or the supplied default.
    """

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def apply_aux_loss_free_router(model: nn.Module, alf_config: Any | None = None) -> dict[str, Any]:
    """Apply auxiliary-loss-free router patching from an ALF config object.

    Args:
        model: Hugging Face Qwen3 MoE model or compatible module tree to patch
            in place.
        alf_config: Mapping, dataclass, or object exposing ALF fields. Supported
            fields are `enabled`, `bias_init`, `bias_update_rate`,
            `bias_update_topk`, `bias_update_schedule`, `bias_update_schedule_steps`,
            `bias_update_end_rate`, `update_interval`, `bias_clip`, `warmup_steps`,
            and `disable_router_aux_loss`.

    Returns:
        A serializable summary with both replacement and patching aliases.
    """

    summary = replace_qwen3_moe_routers(
        model,
        alf_enabled=bool(_config_value(alf_config, "enabled", True)),
        expert_bias_init=float(_config_value(alf_config, "bias_init", 0.0)),
        expert_bias_update_rate=float(_config_value(alf_config, "bias_update_rate", 0.0)),
        expert_bias_update_policy=str(_config_value(alf_config, "bias_update_policy", "proportional")),
        expert_bias_update_interval=int(_config_value(alf_config, "update_interval", 1)),
        expert_bias_ema_beta=float(_config_value(alf_config, "bias_ema_beta", 0.9)),
        expert_bias_update_topk=int(_config_value(alf_config, "bias_update_topk", 1)),
        expert_bias_update_schedule=str(_config_value(alf_config, "bias_update_schedule", "constant")),
        expert_bias_update_schedule_steps=_config_value(alf_config, "bias_update_schedule_steps", None),
        expert_bias_update_end_rate=float(_config_value(alf_config, "bias_update_end_rate", 0.0)),
        expert_bias_clip=_config_value(alf_config, "bias_clip", None),
        expert_bias_warmup_steps=int(_config_value(alf_config, "warmup_steps", 0)),
        disable_original_router_aux_loss=bool(_config_value(alf_config, "disable_router_aux_loss", True)),
    )
    summary["patched_routers"] = summary["num_replaced"]
    summary["patched_module_names"] = tuple(summary["replaced_router_paths"])
    return summary


def build_model_and_tokenizer(model_config: ModelConfig, alf_config: AlfConfig) -> tuple[nn.Module, Any]:
    """Build a Qwen3 MoE model and tokenizer for training.

    Args:
        model_config: Model construction options.
        alf_config: Auxiliary-loss-free router options.

    Returns:
        Tuple of model and tokenizer.

    Raises:
        ImportError: If required Transformers classes are unavailable.
        ValueError: If a non-tiny model source is missing.
    """

    if model_config.use_tiny_config:
        if Qwen3MoeConfig is None or Qwen3MoeForCausalLM is None:
            raise ImportError("Transformers Qwen3 MoE classes are required for tiny config.")
        config = Qwen3MoeConfig(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            moe_intermediate_size=model_config.intermediate_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            num_experts=model_config.num_experts,
            num_experts_per_tok=model_config.num_experts_per_tok,
            output_router_logits=not alf_config.enabled,
            router_aux_loss_coef=0.0 if alf_config.enabled else 0.001,
            max_position_embeddings=512,
        )
        model = Qwen3MoeForCausalLM(config)
        model.to(dtype=_torch_dtype(model_config.torch_dtype))
        tokenizer = _load_tokenizer_for_tiny_model(model_config)
    else:
        if model_config.model_name_or_path is None:
            raise ValueError("model.model_name_or_path is required when use_tiny_config is False.")
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("Transformers is required to load Hugging Face models.")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=_torch_dtype(model_config.torch_dtype),
            trust_remote_code=model_config.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_name_or_path or model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
        )

    if alf_config.enabled:
        apply_aux_loss_free_router(model, alf_config)
    else:
        attach_router_load_tracking(model)
    return model, tokenizer


def _load_tokenizer_for_tiny_model(model_config: ModelConfig) -> Any:
    """Load the configured tokenizer for a tiny model, or use a byte fallback."""

    if model_config.tokenizer_name_or_path is None:
        return SimpleByteTokenizer(model_config.vocab_size)
    if AutoTokenizer is None:
        raise ImportError("Transformers is required to load model.tokenizer_name_or_path.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer_size = len(tokenizer)
    if tokenizer_size != int(model_config.vocab_size):
        raise ValueError(
            "Configured tokenizer vocabulary size does not match model.vocab_size: "
            f"len(tokenizer)={tokenizer_size}, vocab_size={model_config.vocab_size}."
        )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def attach_router_load_tracking(model: nn.Module) -> dict[str, Any]:
    """Attach load-tracking hooks to compatible non-ALF Qwen3 routers.

    Args:
        model: Model containing Qwen3 MoE routers.

    Returns:
        Summary of tracked router paths.
    """

    tracked_router_paths: list[str] = []
    for module_name, module in model.named_modules():
        if not is_qwen3_moe_router(module) or isinstance(module, Qwen3MoeAuxiliaryLossFreeTopKRouter):
            continue
        if hasattr(module, "last_expert_load"):
            continue
        module.register_buffer("last_expert_load", torch.zeros(int(module.num_experts), dtype=torch.long))
        module.register_buffer("last_load_fraction", torch.zeros(int(module.num_experts), dtype=torch.float32))
        module.register_forward_hook(_record_plain_router_load_hook)
        tracked_router_paths.append(module_name)
    return {"num_tracked": len(tracked_router_paths), "tracked_router_paths": tracked_router_paths}


def _record_plain_router_load_hook(
    module: nn.Module,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Record expert load from a plain Qwen3 router forward hook.

    Args:
        module: Router module.
        inputs: Forward inputs, unused.
        output: Router output tuple containing selected expert indices.
    """

    del inputs
    if not isinstance(output, tuple) or len(output) < 3:
        return
    router_indices = output[2]
    if not torch.is_tensor(router_indices) or not hasattr(module, "last_expert_load"):
        return
    with torch.no_grad():
        expert_load = torch.bincount(router_indices.reshape(-1), minlength=int(module.num_experts))
        module.last_expert_load.copy_(expert_load.to(device=module.last_expert_load.device, dtype=torch.long))
        total_assignments = int(expert_load.sum().item())
        if total_assignments == 0:
            module.last_load_fraction.zero_()
            return
        load_fraction = expert_load.to(dtype=torch.float32) / float(total_assignments)
        module.last_load_fraction.copy_(load_fraction.to(device=module.last_load_fraction.device))


def load_model_for_inspection(checkpoint: Path) -> nn.Module:
    """Load a checkpoint for router inspection.

    Args:
        checkpoint: Checkpoint directory produced by ``alf-train``.

    Returns:
        Loaded model.

    Raises:
        FileNotFoundError: If no supported weights are found.
    """

    checkpoint_config = checkpoint / "alf_experiment_config.json"
    parent_config = checkpoint.parent / "config.json"
    config_path = checkpoint_config if checkpoint_config.exists() else parent_config
    if config_path.exists():
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = _model_config_from_dict(raw_config["model"])
        alf_config = _alf_config_from_dict(raw_config["alf"])
        model, _ = build_model_and_tokenizer(model_config, alf_config)
        state_dict = _load_checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    if AutoModelForCausalLM is None:
        raise ImportError("Transformers is required to inspect checkpoints without experiment metadata.")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.eval()
    return model


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a torch dtype from a user-facing name.

    Args:
        name: Dtype name.

    Returns:
        Torch dtype.
    """

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(name, torch.float32)


def _model_config_from_dict(values: Mapping[str, Any]) -> ModelConfig:
    """Create ``ModelConfig`` from serialized values.

    Args:
        values: Serialized model config.

    Returns:
        Model config instance.
    """

    allowed = ModelConfig.__dataclass_fields__
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed})


def _alf_config_from_dict(values: Mapping[str, Any]) -> AlfConfig:
    """Create ``AlfConfig`` from serialized values.

    Args:
        values: Serialized ALF config.

    Returns:
        ALF config instance.
    """

    allowed = AlfConfig.__dataclass_fields__
    return AlfConfig(**{key: value for key, value in values.items() if key in allowed})


def _load_checkpoint_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Load model weights from a saved checkpoint directory.

    Args:
        checkpoint: Checkpoint directory.

    Returns:
        Model state dictionary.

    Raises:
        FileNotFoundError: If no supported weight file exists.
    """

    safetensors_path = checkpoint / "model.safetensors"
    pytorch_path = checkpoint / "pytorch_model.bin"
    if safetensors_path.exists():
        return load_safetensors_file(safetensors_path)
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {checkpoint}")


__all__ = [
    "apply_aux_loss_free_router",
    "attach_router_load_tracking",
    "build_model_and_tokenizer",
    "disable_router_aux_loss",
    "is_qwen3_moe_router",
    "iter_auxiliary_loss_free_routers",
    "load_model_for_inspection",
    "replace_qwen3_moe_routers",
]
