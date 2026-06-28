"""Tests for W&B experiment logging helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alf.wandb_logging import ExperimentLogger


class FakeRun:
    """Minimal fake W&B run used by logger tests.

    Attributes:
        logs: Captured ``run.log`` calls.
        artifacts: Captured ``run.log_artifact`` calls.
        summary: Mutable summary dictionary.
        finished: Whether ``finish`` was called.
    """

    def __init__(self) -> None:
        """Create an empty fake run."""

        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[tuple[Any, list[str] | None]] = []
        self.summary: dict[str, Any] = {}
        self.finished = False

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        """Capture logged values.

        Args:
            values: Values passed by the logger.
            step: Optional training step.
        """

        self.logs.append((values, step))

    def log_artifact(self, artifact: Any, aliases: list[str] | None = None) -> None:
        """Capture artifact logging.

        Args:
            artifact: Fake artifact.
            aliases: Optional aliases.
        """

        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        """Mark the run as finished."""

        self.finished = True


class FakeWandb:
    """Minimal fake W&B module used by logger tests.

    Attributes:
        init_kwargs: Captured ``wandb.init`` keyword arguments.
        run: Fake run returned by ``init``.
    """

    def __init__(self) -> None:
        """Create a fake W&B module."""

        self.init_kwargs: dict[str, Any] | None = None
        self.run = FakeRun()

    def init(self, **kwargs: Any) -> FakeRun:
        """Capture init kwargs and return the fake run.

        Args:
            kwargs: Keyword arguments passed to ``wandb.init``.

        Returns:
            Fake W&B run.
        """

        self.init_kwargs = kwargs
        return self.run

    class Table:
        """Fake W&B table.

        Attributes:
            columns: Table columns.
            data: Table rows.
        """

        def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
            """Store table data.

            Args:
                columns: Table column names.
                data: Table row data.
            """

            self.columns = columns
            self.data = data

    class Artifact:
        """Fake W&B artifact.

        Attributes:
            name: Artifact name.
            type: Artifact type.
            metadata: Artifact metadata.
            files: Added file paths.
            dirs: Added directory paths.
        """

        def __init__(self, name: str, type: str, metadata: dict[str, Any] | None = None) -> None:
            """Create a fake artifact.

            Args:
                name: Artifact name.
                type: Artifact type.
                metadata: Artifact metadata.
            """

            self.name = name
            self.type = type
            self.metadata = metadata
            self.files: list[str] = []
            self.dirs: list[str] = []

        def add_file(self, path: str) -> None:
            """Capture an added file path.

            Args:
                path: File path.
            """

            self.files.append(path)

        def add_dir(self, path: str) -> None:
            """Capture an added directory path.

            Args:
                path: Directory path.
            """

            self.dirs.append(path)


def test_disabled_logger_does_not_import_or_call_wandb(monkeypatch: Any) -> None:
    """Disabled mode should be a no-op and never import W&B."""

    def fail_on_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise if the logger attempts to import W&B.

        Args:
            name: Module name.
            args: Positional import arguments.
            kwargs: Keyword import arguments.

        Returns:
            Imported module for non-W&B imports.

        Raises:
            AssertionError: If W&B is imported.
        """

        if name == "wandb":
            raise AssertionError("disabled logger imported wandb")
        return original_import_module(name, *args, **kwargs)

    original_import_module = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", fail_on_wandb_import)
    config = SimpleNamespace(enabled=False, entity=None, project=None, mode=None, group=None, tags=None)

    logger = ExperimentLogger(config)
    logger.log({"loss": 1.0}, step=1)
    logger.log_expert_activation_table("experts", [{"expert": 0, "load": 3}], step=1)
    logger.log_artifact(Path("missing"))
    logger.update_summary({"best_loss": 1.0})
    logger.finish()

    assert logger.enabled is False
    assert logger.run is None


def test_wandb_disabled_mode_does_not_import_or_call_wandb(monkeypatch: Any) -> None:
    """W&B mode disabled should also be a no-op without importing W&B."""

    def fail_on_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise if the logger attempts to import W&B.

        Args:
            name: Module name.
            args: Positional import arguments.
            kwargs: Keyword import arguments.

        Returns:
            Imported module for non-W&B imports.

        Raises:
            AssertionError: If W&B is imported.
        """

        if name == "wandb":
            raise AssertionError("disabled logger imported wandb")
        return original_import_module(name, *args, **kwargs)

    original_import_module = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", fail_on_wandb_import)
    config = SimpleNamespace(
        enabled=True,
        entity="entity",
        project="project",
        mode="disabled",
        group=None,
        tags=None,
    )

    logger = ExperimentLogger(config)
    logger.log({"loss": 1.0}, step=1)
    logger.finish()

    assert logger.enabled is False
    assert logger.run is None


def test_env_defaults_and_stable_log_keys(monkeypatch: Any) -> None:
    """Enabled logger should read env defaults and flatten nested metric keys."""

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_RUN_GROUP", "router-ablation")
    monkeypatch.setenv("WANDB_TAGS", "alf, tiny")
    config = SimpleNamespace(
        enabled=True,
        entity=None,
        project=None,
        mode=None,
        group=None,
        tags=None,
        log_checkpoints=True,
    )

    logger = ExperimentLogger(config, experiment_name="run-a", config={"seed": 7})
    logger.log(
        {
            "router": {
                "aggregate_load": {
                    "mean_load": 2.0,
                    "counts": [1, 3],
                }
            },
            "loss": 0.5,
        },
        step=4,
    )
    logger.update_summary({"best_loss": 0.5})
    logger.finish()

    assert fake_wandb.init_kwargs == {
        "entity": "test-entity",
        "project": "test-project",
        "mode": "offline",
        "group": "router-ablation",
        "tags": ["alf", "tiny"],
        "name": "run-a",
        "config": {"seed": 7},
    }
    assert fake_wandb.run.logs == [
        (
            {
                "loss": 0.5,
                "router/aggregate_load/counts": [1, 3],
                "router/aggregate_load/mean_load": 2.0,
            },
            4,
        )
    ]
    assert fake_wandb.run.summary == {"best_loss": 0.5}
    assert fake_wandb.run.finished is True
    assert logger.enabled is False


def test_online_mode_requires_entity_and_project(monkeypatch: Any) -> None:
    """Online mode should fail fast without explicit W&B destination."""

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    config = SimpleNamespace(
        enabled=True,
        entity=None,
        project=None,
        mode="online",
        group=None,
        tags=None,
        log_checkpoints=True,
    )

    try:
        ExperimentLogger(config)
    except ValueError as exc:
        assert "WANDB_ENTITY" in str(exc)
    else:
        raise AssertionError("online W&B without entity/project should fail")


def test_tables_and_checkpoint_artifacts_use_stable_names(monkeypatch: Any, tmp_path: Path) -> None:
    """Tables and artifacts should use deterministic keys and respect checkpoint config."""

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = {
        "enabled": True,
        "entity": "entity",
        "project": "project",
        "mode": "offline",
        "group": None,
        "tags": ["alf"],
        "log_checkpoints": False,
    }
    artifact_file = tmp_path / "metrics.json"
    artifact_file.write_text("{}", encoding="utf-8")

    logger = ExperimentLogger(config)
    logger.log_expert_activation_table("experts", [{"load": 3, "expert": 0}], step=2)
    logger.log_artifact(artifact_file, artifact_type="checkpoint", aliases=["latest"])
    logger.log_artifact(artifact_file, name="metrics", artifact_type="metrics", aliases=["latest"])

    logged_table = fake_wandb.run.logs[0][0]["experts/table"]
    assert logged_table.columns == ["expert", "load"]
    assert logged_table.data == [[0, 3]]
    assert len(fake_wandb.run.artifacts) == 1
    artifact, aliases = fake_wandb.run.artifacts[0]
    assert artifact.name == "metrics"
    assert artifact.type == "metrics"
    assert artifact.files == [str(artifact_file)]
    assert aliases == ["latest"]
