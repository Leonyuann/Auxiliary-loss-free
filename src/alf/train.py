"""Training entry point for ALF causal language model experiments."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from alf.config import asdict, load_experiment_config, parse_config_args
from alf.data import build_packed_text_dataset, causal_lm_collate
from alf.metrics import append_jsonl, collect_router_metrics
from alf.modeling import build_model_and_tokenizer


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
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

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
    scheduler = _build_scheduler(optimizer, config.training.learning_rate, config.training.warmup_steps)

    start_step = 0
    if config.training.resume_from:
        start_step = _load_checkpoint(Path(config.training.resume_from), model, optimizer, scheduler)

    metrics_path = output_dir / "metrics.jsonl"
    step = start_step
    progress = tqdm(total=config.training.max_steps, initial=start_step, desc=config.name)
    data_iter = _cycle(loader)
    last_checkpoint = output_dir / "latest"

    while step < config.training.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        total_loss = 0.0
        tokens = 0

        for _ in range(config.training.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.detach().float().item()
            tokens += int(batch["input_ids"].numel())

        optimizer.step()
        scheduler.step()
        step += 1
        progress.update(1)

        elapsed = max(time.perf_counter() - step_start, 1e-9)
        if step % config.training.log_every == 0 or step == config.training.max_steps:
            record: dict[str, Any] = {
                "step": step,
                "loss": total_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "tokens_per_second": tokens / elapsed,
                "router": collect_router_metrics(model),
            }
            append_jsonl(metrics_path, record)
            progress.set_postfix(loss=f"{total_loss:.4f}")

        if step % config.training.save_every == 0 or step == config.training.max_steps:
            last_checkpoint = _save_checkpoint(output_dir, "latest", model, optimizer, scheduler, step, asdict(config))

    progress.close()
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
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear warmup scheduler.

    Args:
        optimizer: Optimizer to schedule.
        learning_rate: Target learning rate.
        warmup_steps: Number of warmup steps.

    Returns:
        LambdaLR scheduler.
    """

    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    def lr_lambda(step: int) -> float:
        """Return warmup LR multiplier.

        Args:
            step: Scheduler step index.

        Returns:
            Learning-rate multiplier.
        """

        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

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
