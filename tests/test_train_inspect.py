"""Integration tests for training checkpoint and inspection flows."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

import torch
from safetensors.torch import load_file

from alf.inspect import inspect_router
from alf.train import _build_scheduler, train


def test_train_resume_restores_model_state(tmp_path: Path) -> None:
    """Resume should restore saved model weights before continuing training."""

    output_dir = tmp_path / "resume"
    checkpoint = train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )
    saved_state = load_file(checkpoint / "model.safetensors")
    saved_key = next(key for key in saved_state if key.endswith("mlp.gate.weight"))
    saved_weight = saved_state[saved_key].clone()

    train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.resume_from",
            str(checkpoint),
            "--wandb.enabled",
            "false",
        ],
    )
    resumed_state = load_file(checkpoint / "model.safetensors")

    assert torch.allclose(resumed_state[saved_key], saved_weight)


def test_aux_loss_baseline_records_router_load_metrics(tmp_path: Path) -> None:
    """The auxiliary-loss baseline should expose comparable router load metrics."""

    output_dir = tmp_path / "aux"
    checkpoint = train(
        "experiments/qwen3_moe_tiny_aux_loss.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )

    metrics = inspect_router(checkpoint)

    assert metrics["num_routers"] == 2
    assert "aggregate_load" in metrics
    assert metrics["aggregate_load"]["total_assignments"] > 0


def test_checkpoint_inspection_is_self_contained_after_copy(tmp_path: Path) -> None:
    """Copied checkpoint directories should still inspect ALF router metrics."""

    output_dir = tmp_path / "source"
    checkpoint = train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )
    copied_checkpoint = tmp_path / "copied-latest"
    shutil.copytree(checkpoint, copied_checkpoint)

    metrics = inspect_router(copied_checkpoint)

    assert metrics["num_routers"] == 2
    assert "aggregate_bias" in metrics
    assert metrics["aggregate_load"]["total_assignments"] > 0


def test_training_metrics_include_wandb_observability_keys(tmp_path: Path) -> None:
    """Training JSONL should contain the metrics mirrored to W&B."""

    output_dir = tmp_path / "observability"
    train(
        "experiments/qwen3_moe_tiny_aux_loss.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--eval.eval_every",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )

    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    train_record = next(record for record in records if "train" in record)
    eval_record = next(record for record in records if "eval/ppl" in record)
    assert set(train_record["train"]) >= {
        "loss",
        "lm_loss",
        "aux_loss",
        "aux_loss_scaled",
        "learning_rate",
        "grad_norm",
        "tokens_per_second",
        "maxvio_batch",
        "maxvio_batch_rolling_100",
    }
    assert eval_record["eval/ppl"] > 0.0
    assert eval_record["eval/maxvio_global"] >= 0.0
    assert train_record["expert_activation"]["train"]["matrix"]["layers"]
    assert train_record["expert_activation"]["train"]["rows"]
    assert eval_record["expert_activation"]["eval"]["matrix"]["layers"]
    assert eval_record["expert_activation"]["eval"]["rows"]


def test_alf_bias_updates_once_per_optimizer_step_with_accumulation(tmp_path: Path) -> None:
    """ALF bias updates should happen once per router after gradient accumulation."""

    output_dir = tmp_path / "alf-accumulation"
    train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "1",
            "--training.gradient_accumulation_steps",
            "2",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )

    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_record = next(record for record in records if "train" in record)

    assert train_record["train"]["bias_update_events"] == 2


def test_validation_files_are_used_for_eval_metrics(tmp_path: Path) -> None:
    """Validation should use explicit validation files and validation sample limits."""

    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "validation.txt"
    train_file.write_text("train token " * 200, encoding="utf-8")
    val_file.write_text("validation token " * 40, encoding="utf-8")
    output_dir = tmp_path / "validation-flow"

    train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--data.train_files",
            str(train_file),
            "--data.validation_files",
            str(val_file),
            "--data.max_train_samples",
            "2",
            "--data.max_validation_samples",
            "3",
            "--eval.max_eval_samples",
            "1",
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--eval.eval_every",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )

    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eval_record = next(record for record in records if "eval/tokens" in record)

    assert eval_record["eval/tokens"] == 32


def test_train_module_entrypoint_runs_main(tmp_path: Path) -> None:
    """Running python -m alf.train should execute the training entry point."""

    output_dir = tmp_path / "module-entry"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alf.train",
            "experiments/qwen3_moe_tiny_alf.py",
            "--training.max_steps",
            "0",
            "--training.output_dir",
            str(output_dir),
            "--wandb.enabled",
            "false",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "config.json").exists()


def test_cosine_scheduler_decays_after_warmup() -> None:
    """Cosine scheduler should warm up and then anneal toward zero."""

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = _build_scheduler(optimizer, 0.1, 1, max_steps=4, scheduler_type="cosine")

    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert lrs[0] > lrs[-1]
    assert lrs[-1] == 0.0
