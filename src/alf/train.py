"""Training entry point for ALF causal language model experiments."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from alf.config import asdict, load_experiment_config, parse_config_args
from alf.data import build_packed_text_dataset, causal_lm_collate
from alf.eval import evaluate_model
from alf.metrics import (
    activation_matrix_from_counts,
    activation_rows_from_counts,
    add_bias_update_deltas,
    add_layer_counts,
    append_jsonl,
    bias_update_matrix_from_deltas,
    bias_update_rows_from_deltas,
    collect_bias_update_deltas,
    collect_bias_update_steps,
    collect_expert_load_counts,
    collect_router_metrics,
    loss_breakdown,
    mean_maxvio,
    serialize_activation_matrix,
)
from alf.modeling import build_model_and_tokenizer
from alf.wandb_logging import ExperimentLogger


def train(config_path: str | Path, overrides: list[str] | None = None) -> Path:
    """Train one causal language model experiment.

    Args:
        config_path: Python experiment config path.
        overrides: Optional dotted CLI overrides.

    Returns:
        Path to the latest checkpoint directory.
    """

    config = load_experiment_config(config_path, overrides)
    _set_seed(config.training.seed)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    logger = ExperimentLogger(config.wandb, experiment_name=config.name, config=config_dict)

    device = _resolve_device(config.training.device)
    model, tokenizer = build_model_and_tokenizer(config.model, config.alf)
    model.to(device)
    model.train()

    dataset = build_packed_text_dataset(
        tokenizer=tokenizer,
        paths=config.data.train_files,
        block_size=config.data.block_size,
        max_train_samples=config.data.max_train_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=causal_lm_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        config.training.learning_rate,
        config.training.warmup_steps,
        max_steps=config.training.max_steps,
        scheduler_type=config.training.scheduler_type,
    )

    start_step = 0
    if config.training.resume_from:
        start_step = _load_checkpoint(Path(config.training.resume_from), model, optimizer, scheduler)

    router_bias_update_steps = collect_bias_update_steps(model)
    metrics_path = output_dir / "metrics.jsonl"
    step = start_step
    progress = tqdm(total=config.training.max_steps, initial=start_step, desc=config.name)
    data_iter = _cycle(loader)
    last_checkpoint = output_dir / "latest"
    maxvio_window: deque[float] = deque(maxlen=100)
    best_eval_ppl: float | None = None
    best_eval_maxvio: float | None = None

    try:
        while step < config.training.max_steps:
            optimizer.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            loss_totals = {"loss": 0.0, "lm_loss": 0.0, "aux_loss": 0.0, "aux_loss_scaled": 0.0}
            tokens = 0
            step_layer_counts: dict[str, torch.Tensor] = {}
            step_bias_deltas: dict[str, torch.Tensor] = {}
            bias_update_events = 0

            for _ in range(config.training.gradient_accumulation_steps):
                batch = next(data_iter)
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                breakdown = loss_breakdown(outputs, model)
                loss = outputs.loss / config.training.gradient_accumulation_steps
                loss.backward()
                for key in loss_totals:
                    loss_totals[key] += breakdown[key] / config.training.gradient_accumulation_steps
                tokens += int(batch["input_ids"].numel())
                add_layer_counts(step_layer_counts, collect_expert_load_counts(model))
                bias_deltas, update_events = collect_bias_update_deltas(model, router_bias_update_steps)
                add_bias_update_deltas(step_bias_deltas, bias_deltas)
                bias_update_events += update_events

            grad_norm = _gradient_norm(model)
            optimizer.step()
            scheduler.step()
            step += 1
            progress.update(1)

            elapsed = max(time.perf_counter() - step_start, 1e-9)
            maxvio_batch = mean_maxvio(step_layer_counts)
            activation_matrix, activation_layers = activation_matrix_from_counts(step_layer_counts)
            activation_matrix_json = serialize_activation_matrix(activation_matrix, activation_layers)
            activation_rows = activation_rows_from_counts(step_layer_counts, step=step, split="train")
            bias_update_matrix, bias_update_layers = bias_update_matrix_from_deltas(step_bias_deltas)
            bias_update_rows = bias_update_rows_from_deltas(step_bias_deltas, step=step)

            record: dict[str, Any] = {
                "step": step,
                "train": {
                    "loss": loss_totals["loss"],
                    "lm_loss": loss_totals["lm_loss"],
                    "aux_loss": loss_totals["aux_loss"],
                    "aux_loss_scaled": loss_totals["aux_loss_scaled"],
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "grad_norm": grad_norm,
                    "tokens_per_second": tokens / elapsed,
                    "maxvio_batch": maxvio_batch,
                    "bias_update_events": bias_update_events,
                },
                "router": collect_router_metrics(model),
                "expert_activation": {
                    "train": {
                        "matrix": activation_matrix_json,
                        "rows": activation_rows,
                    }
                },
            }
            if bias_update_events > 0:
                record["bias_update"] = {
                    "train": {
                        "events": bias_update_events,
                        "matrix": serialize_activation_matrix(bias_update_matrix, bias_update_layers),
                        "rows": bias_update_rows,
                    }
                }

            should_log_step = step % config.training.log_every == 0 or step == config.training.max_steps
            if should_log_step:
                maxvio_window.append(maxvio_batch)
                record["train"]["maxvio_batch_rolling_100"] = float(sum(maxvio_window) / len(maxvio_window))
                append_jsonl(metrics_path, record)
                logger.log(record, step=step)
                logger.log_expert_activation_heatmap("train/expert_activation", activation_matrix, step=step)
                logger.log_expert_activation_table("train/expert_activation", activation_rows, step=step)
                if bias_update_events > 0:
                    logger.log_bias_update_heatmap("train/bias_update", bias_update_matrix, step=step)
                    logger.log_expert_activation_table("train/bias_update", bias_update_rows, step=step)
                progress.set_postfix(loss=f"{loss_totals['loss']:.4f}")

            if _should_evaluate(step, config.eval.eval_every, config.training.max_steps):
                eval_record = evaluate_model(model, tokenizer, config, device)
                eval_scalars = {
                    key: value
                    for key, value in eval_record.items()
                    if not key.endswith("_matrix")
                    and not key.endswith("_rows")
                    and not key.endswith("_layers")
                    and not key.endswith("_matrix_json")
                }
                eval_json_record = {
                    "step": step,
                    **eval_scalars,
                    "expert_activation": {
                        "eval": {
                            "matrix": eval_record["eval/expert_activation_matrix_json"],
                            "rows": eval_record["eval/expert_activation_rows"],
                        }
                    },
                }
                append_jsonl(metrics_path, eval_json_record)
                logger.log(eval_scalars, step=step)
                logger.log_expert_activation_heatmap(
                    "eval/expert_activation",
                    eval_record["eval/expert_activation_matrix"],
                    step=step,
                )
                logger.log_expert_activation_table(
                    "eval/expert_activation",
                    eval_record["eval/expert_activation_rows"],
                    step=step,
                )
                eval_ppl = float(eval_record["eval/ppl"])
                eval_maxvio = float(eval_record["eval/maxvio_global"])
                best_eval_ppl = eval_ppl if best_eval_ppl is None else min(best_eval_ppl, eval_ppl)
                best_eval_maxvio = eval_maxvio if best_eval_maxvio is None else min(best_eval_maxvio, eval_maxvio)
                logger.update_summary(
                    {
                        "best/eval_ppl": best_eval_ppl,
                        "best/eval_maxvio_global": best_eval_maxvio,
                        "final/eval_ppl": eval_ppl,
                        "final/eval_maxvio_global": eval_maxvio,
                    }
                )

            if step % config.training.save_every == 0 or step == config.training.max_steps:
                last_checkpoint = _save_checkpoint(output_dir, "latest", model, optimizer, scheduler, step, asdict(config))
                logger.log_artifact(
                    last_checkpoint,
                    name=f"{config.name}-latest",
                    artifact_type="checkpoint",
                    aliases=["latest", f"step-{step}"],
                    metadata={"step": step},
                )

    finally:
        progress.close()
        logger.finish()
    return last_checkpoint


def main() -> None:
    """Run the command-line training entry point."""

    config_path, overrides = parse_config_args()
    train(config_path, overrides)


def _resolve_device(device: str) -> torch.device:
    """Resolve a device string.

    Args:
        device: User-provided device string.

    Returns:
        Torch device.
    """

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    """Seed Python and Torch RNGs.

    Args:
        seed: Random seed.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gradient_norm(model: torch.nn.Module) -> float:
    """Compute global L2 gradient norm without modifying gradients.

    Args:
        model: Model whose gradients should be measured.

    Returns:
        Gradient norm.
    """

    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
    return float(total**0.5)


def _should_evaluate(step: int, eval_every: int, max_steps: int) -> bool:
    """Return whether validation should run at a step.

    Args:
        step: Current training step.
        eval_every: Evaluation interval. Zero disables periodic eval.
        max_steps: Final training step.

    Returns:
        Whether to evaluate.
    """

    if eval_every <= 0:
        return step == max_steps
    return step % eval_every == 0 or step == max_steps


def _cycle(loader: DataLoader) -> Any:
    """Yield batches from a dataloader forever.

    Args:
        loader: Source dataloader.

    Yields:
        Batches from the dataloader.
    """

    while True:
        for batch in loader:
            yield batch


def _save_checkpoint(
    output_dir: Path,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config_dict: dict[str, Any],
) -> Path:
    """Save a training checkpoint.

    Args:
        output_dir: Experiment output directory.
        name: Checkpoint directory name.
        model: Model to save.
        optimizer: Optimizer to save.
        scheduler: Scheduler to save.
        step: Current training step.
        config_dict: Serialized experiment config.

    Returns:
        Checkpoint directory path.
    """

    checkpoint_dir = output_dir / name
    tmp_dir = output_dir / f".{name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    model.save_pretrained(tmp_dir)
    (tmp_dir / "alf_experiment_config.json").write_text(
        json.dumps(config_dict, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step},
        tmp_dir / "trainer_state.pt",
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)
    return checkpoint_dir


def _load_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Load model and optimizer state from a checkpoint directory.

    Args:
        checkpoint_dir: Checkpoint directory.
        model: Model to update.
        optimizer: Optimizer to update.
        scheduler: Scheduler to update.

    Returns:
        Restored training step.
    """

    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        return 0
    model_state = _load_model_state_dict(checkpoint_dir)
    model.load_state_dict(model_state, strict=False)
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    warmup_steps: int,
    *,
    max_steps: int | None = None,
    scheduler_type: str = "constant",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a warmup scheduler with optional cosine annealing."""

    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    normalized_type = scheduler_type.lower().replace("-", "_")
    if normalized_type not in {"constant", "cosine", "cosine_annealing"}:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type!r}")
    if normalized_type in {"cosine", "cosine_annealing"} and max_steps is None:
        raise ValueError("max_steps is required for cosine scheduler.")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if normalized_type == "constant":
            return 1.0
        assert max_steps is not None
        decay_steps = max(1, int(max_steps) - int(warmup_steps))
        progress = min(1.0, max(0.0, float(step - warmup_steps + 1) / float(decay_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _load_model_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load saved model weights from a checkpoint directory.

    Args:
        checkpoint_dir: Checkpoint directory.

    Returns:
        Model state dictionary.

    Raises:
        FileNotFoundError: If no supported weight file exists.
    """

    safetensors_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(safetensors_path)
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
