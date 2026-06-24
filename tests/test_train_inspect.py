"""Integration tests for training checkpoint and inspection flows."""

from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file

from alf.inspect import inspect_router
from alf.train import train


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
        ],
    )
    copied_checkpoint = tmp_path / "copied-latest"
    shutil.copytree(checkpoint, copied_checkpoint)

    metrics = inspect_router(copied_checkpoint)

    assert metrics["num_routers"] == 2
    assert "aggregate_bias" in metrics
    assert metrics["aggregate_load"]["total_assignments"] > 0
