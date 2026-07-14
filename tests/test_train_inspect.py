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
    assert "system" in train_record
    assert set(train_record["system"]) >= {
        "step_time_ms",
        "step_time_ms_rolling_100",
        "tokens_per_sec",
        "tokens_per_sec_rolling_100",
        "gpu_memory_allocated",
    }
    assert "moe" in train_record
    assert set(train_record["moe"]) >= {
        "expert_load_max_over_mean",
        "expert_load_cv",
        "expert_load_normalized_entropy",
        "overflow_rate",
        "dropped_token_rate",
    }
    assert eval_record["eval/ppl"] > 0.0
    assert eval_record["eval/maxvio_global"] >= 0.0
    assert 0.0 <= eval_record["eval/layerwise_normalized_entropy_mean"] <= 1.0
    assert 0.0 <= eval_record["eval/layerwise_normalized_entropy_min"] <= 1.0
    assert 0.0 <= eval_record["eval/layerwise_normalized_entropy_max"] <= 1.0
    assert train_record["expert_activation"]["train"]["matrix"]["layers"]
    assert train_record["expert_activation"]["train"]["rows"]
    assert eval_record["expert_activation"]["eval"]["matrix"]["layers"]
    assert eval_record["expert_activation"]["eval"]["rows"]
    assert eval_record["layerwise_normalized_entropy"]["eval"]["rows"]


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


def test_adaptive_ema_policies_log_dynamic_router_state(tmp_path: Path) -> None:
    """Adaptive EMA policies should run and expose their dynamic controller state."""

    for policy in [
        "adaptive_ema_variance",
        "adaptive_ema_persistent_oscillation",
        "adaptive_ema_gain_coupled",
    ]:
        output_dir = tmp_path / policy
        train(
            "experiments/qwen3_moe_tiny_alf.py",
            [
                "--training.max_steps",
                "1",
                "--training.output_dir",
                str(output_dir),
                "--training.save_every",
                "1",
                "--alf.bias_update_policy",
                policy,
                "--alf.bias_update_rate",
                "0.1",
                "--wandb.enabled",
                "false",
            ],
        )

        records = [
            json.loads(line)
            for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train_record = next(record for record in records if "train" in record)
        router_summaries = train_record["router"]["routers"].values()

        for summary in router_summaries:
            assert summary["bias_update_policy"] == policy
            assert 0.1 <= summary["adaptive_ema_beta"] <= 0.95
            assert summary["normalized_load_variance"] >= 0.0
            assert summary["load_batch_noise"] > 0.0
            assert summary["normalized_feedback_gain"] >= 0.0
            assert summary["gain_coupled_normalized_gain"] > 0.0


def test_adaptive_per_expert_policy_runs_tiny_train_and_logs_state(tmp_path: Path) -> None:
    """Tiny training should update and expose per-expert second-moment state."""

    output_dir = tmp_path / "adaptive-per-expert"
    checkpoint = train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--alf.bias_update_policy",
            "adaptive_per_expert",
            "--alf.bias_update_rate",
            "0.001",
            "--wandb.enabled",
            "false",
        ],
    )
    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_record = next(record for record in records if "train" in record)

    for summary in train_record["router"]["routers"].values():
        assert summary["bias_update_policy"] == "adaptive_per_expert"
        assert summary["load_error_second_moment"]["max"] >= 0.0
        assert summary["effective_update_rate"]["max"] > 0.0

    inspected = inspect_router(checkpoint)
    for summary in inspected["routers"].values():
        assert summary["load_error_second_moment"]["values"]
        assert summary["effective_update_rate"]["values"]


def test_adaptive_per_expert_momentum_policy_logs_state(tmp_path: Path) -> None:
    """Tiny training should checkpoint and expose the momentum controller state."""

    output_dir = tmp_path / "adaptive-per-expert-momentum"
    checkpoint = train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "1",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "1",
            "--alf.bias_update_policy",
            "adaptive_per_expert_momentum",
            "--alf.bias_update_rate",
            "0.001",
            "--wandb.enabled",
            "false",
        ],
    )
    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_record = next(record for record in records if "train" in record)

    for summary in train_record["router"]["routers"].values():
        assert summary["bias_update_policy"] == "adaptive_per_expert_momentum"
        assert summary["load_error_momentum"]["values"]
        assert summary["load_error_second_moment"]["max"] >= 0.0
        assert summary["effective_update_rate"]["max"] > 0.0

    inspected = inspect_router(checkpoint)
    for summary in inspected["routers"].values():
        assert summary["load_error_momentum"]["values"]


def test_alf_bias_max_update_step_freezes_training_updates(tmp_path: Path) -> None:
    """Training should stop reporting bias updates after the configured step."""

    output_dir = tmp_path / "alf-max-bias-step"
    train(
        "experiments/qwen3_moe_tiny_alf.py",
        [
            "--training.max_steps",
            "2",
            "--training.output_dir",
            str(output_dir),
            "--training.save_every",
            "2",
            "--alf.bias_max_update_steps",
            "1",
            "--wandb.enabled",
            "false",
        ],
    )

    train_records = [
        record
        for record in (
            json.loads(line)
            for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if "train" in record
    ]

    assert [record["train"]["bias_update_events"] for record in train_records] == [2, 0]


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
