"""Weights & Biases logging helpers for ALF experiments."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class _NoOpSummary:
    """No-op summary object used when experiment logging is disabled.

    Methods:
        update: Accept summary values without side effects.
    """

    def update(self, values: Mapping[str, Any]) -> None:
        """Ignore summary values.

        Args:
            values: Summary values that would be sent to W&B when enabled.
        """


class ExperimentLogger:
    """Small W&B-backed experiment logger with a disabled no-op mode.

    Attributes:
        enabled: Whether this logger will call the W&B SDK.
        run: Active W&B run object, or ``None`` when disabled.

    Properties:
        summary: W&B run summary or a no-op summary object.

    Methods:
        log: Log scalar and nested metric dictionaries.
        log_expert_activation_heatmap: Log an expert-activation matrix as an image.
        log_expert_activation_table: Log expert-activation rows as a W&B table.
        log_artifact: Log a file or directory as a W&B artifact.
        update_summary: Update run summary values.
        finish: Finish the active W&B run.

    Notes:
        The ``wandb`` module is imported only after the config resolves to an
        enabled non-disabled mode. Disabled runs are therefore safe in
        environments where W&B is not installed or configured.
    """

    def __init__(
        self,
        wandb_config: Any | None = None,
        *,
        experiment_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an experiment logger from a WandbConfig-like object.

        Args:
            wandb_config: Object or mapping with optional ``enabled``, ``entity``,
                ``project``, ``mode``, ``group``, ``tags``, and
                ``log_checkpoints`` fields.
            experiment_name: Optional run name passed to ``wandb.init``.
            config: Optional experiment config dictionary passed to W&B.
        """

        self.enabled = False
        self.run: Any | None = None
        self._wandb: Any | None = None
        self._summary = _NoOpSummary()
        self._log_checkpoints = bool(_get_config_value(wandb_config, "log_checkpoints", True))

        if not bool(_get_config_value(wandb_config, "enabled", False)):
            return

        mode = _resolve_string_field(wandb_config, "mode", "WANDB_MODE", "online")
        if mode.lower() == "disabled":
            return
        entity = _resolve_string_field(wandb_config, "entity", "WANDB_ENTITY", None)
        project = _resolve_string_field(wandb_config, "project", "WANDB_PROJECT", None)
        if mode.lower() == "online" and (not entity or not project):
            msg = "WANDB_ENTITY and WANDB_PROJECT are required when W&B mode is online."
            raise ValueError(msg)

        self._wandb = importlib.import_module("wandb")
        self.run = self._wandb.init(
            entity=entity,
            project=project,
            mode=mode,
            group=_resolve_string_field(wandb_config, "group", "WANDB_RUN_GROUP", None),
            tags=_resolve_tags(wandb_config),
            name=experiment_name,
            config=dict(config) if config is not None else None,
        )
        self.enabled = True
        self._summary = getattr(self.run, "summary", _NoOpSummary())

    @property
    def summary(self) -> Any:
        """Return the active W&B summary object or a no-op replacement.

        Returns:
            Summary-like object exposing ``update``.
        """

        return self._summary

    def log(self, values: Mapping[str, Any], step: int | None = None) -> None:
        """Log a metric dictionary with stable flattened nested keys.

        Args:
            values: Metric values to log. Nested mappings are flattened with
                slash-separated keys, for example ``router/load/mean``.
            step: Optional training step.
        """

        if not self.enabled or self.run is None:
            return
        self.run.log(_flatten_mapping(values), step=step)

    def log_expert_activation_heatmap(self, name: str, matrix: Any, step: int | None = None) -> None:
        """Log an expert activation matrix as a W&B image heatmap.

        Args:
            name: Metric namespace for the image.
            matrix: Two-dimensional matrix-like object accepted by matplotlib.
            step: Optional training step.
        """

        self._log_layer_expert_heatmap(name, matrix, step=step, cmap="viridis")

    def log_bias_update_heatmap(self, name: str, matrix: Any, step: int | None = None) -> None:
        """Log a per-layer, per-expert bias update matrix as a W&B heatmap.

        Args:
            name: Metric namespace for the image.
            matrix: Two-dimensional matrix of expert-bias update deltas.
            step: Optional training step.
        """

        self._log_layer_expert_heatmap(
            name,
            matrix,
            step=step,
            cmap="coolwarm",
            center_zero=True,
            colorbar_label="Bias delta",
        )

    def _log_layer_expert_heatmap(
        self,
        name: str,
        matrix: Any,
        *,
        step: int | None = None,
        cmap: str = "viridis",
        center_zero: bool = False,
        colorbar_label: str | None = None,
    ) -> None:
        """Log a layer-by-expert matrix as a W&B image heatmap."""

        if not self.enabled or self.run is None or self._wandb is None:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import TwoSlopeNorm

        values = np.asarray(_to_python_array(matrix), dtype=float)
        if values.size == 0:
            return

        norm = None
        if center_zero:
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                limit = float(np.abs(finite_values).max())
                if limit > 0.0:
                    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")
        ax.set_title(name)
        colorbar = fig.colorbar(image, ax=ax)
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)
        try:
            self.run.log({f"{name}/heatmap": self._wandb.Image(fig)}, step=step)
        finally:
            plt.close(fig)

    def log_expert_activation_table(
        self,
        name: str,
        rows: Sequence[Mapping[str, Any]] | Sequence[Sequence[Any]],
        step: int | None = None,
    ) -> None:
        """Log expert activation rows as a W&B table.

        Args:
            name: Metric namespace for the table.
            rows: Row dictionaries or row sequences.
            step: Optional training step.
        """

        if not self.enabled or self.run is None or self._wandb is None:
            return

        columns, data = _table_data(rows)
        table = self._wandb.Table(columns=columns, data=data)
        self.run.log({f"{name}/table": table}, step=step)

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        artifact_type: str = "checkpoint",
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Log a file or directory as a W&B artifact.

        Args:
            path: File or directory path to attach.
            name: Optional artifact name. Defaults to the path name.
            artifact_type: W&B artifact type.
            aliases: Optional artifact aliases.
            metadata: Optional artifact metadata.
        """

        if not self.enabled or self.run is None or self._wandb is None:
            return
        if artifact_type == "checkpoint" and not self._log_checkpoints:
            return

        artifact_path = Path(path)
        artifact = self._wandb.Artifact(
            name=name or artifact_path.name,
            type=artifact_type,
            metadata=dict(metadata) if metadata is not None else None,
        )
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        self.run.log_artifact(artifact, aliases=list(aliases) if aliases is not None else None)

    def update_summary(self, values: Mapping[str, Any]) -> None:
        """Update run summary values.

        Args:
            values: Summary values to merge into the W&B run summary.
        """

        self.summary.update(dict(values))

    def finish(self) -> None:
        """Finish the active W&B run if logging is enabled."""

        if not self.enabled or self.run is None:
            return
        self.run.finish()
        self.enabled = False
        self.run = None
        self._wandb = None
        self._summary = _NoOpSummary()


def _get_config_value(config: Any | None, field: str, default: Any) -> Any:
    """Read a field from a mapping or object.

    Args:
        config: Mapping, object, or ``None``.
        field: Field name to read.
        default: Fallback value.

    Returns:
        Field value or the fallback.
    """

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(field, default)
    return getattr(config, field, default)


def _resolve_string_field(config: Any | None, field: str, env_name: str, default: str | None) -> str | None:
    """Resolve a string field from config, environment, then fallback.

    Args:
        config: Mapping, object, or ``None``.
        field: Config field name.
        env_name: Environment variable name.
        default: Fallback value.

    Returns:
        Resolved string or ``None``.
    """

    value = _get_config_value(config, field, None)
    if value is not None:
        return str(value)
    return os.environ.get(env_name, default)


def _resolve_tags(config: Any | None) -> list[str] | None:
    """Resolve W&B tags from config or ``WANDB_TAGS``.

    Args:
        config: Mapping, object, or ``None``.

    Returns:
        List of tags or ``None``.
    """

    value = _get_config_value(config, "tags", None)
    if value is None:
        env_tags = os.environ.get("WANDB_TAGS")
        if env_tags is None:
            return None
        return [tag.strip() for tag in env_tags.split(",") if tag.strip()]
    if isinstance(value, str):
        return [value]
    return [str(tag) for tag in value]


def _flatten_mapping(values: Mapping[str, Any], prefix: str | None = None) -> dict[str, Any]:
    """Flatten nested mappings into slash-separated metric keys.

    Args:
        values: Mapping to flatten.
        prefix: Optional parent key prefix.

    Returns:
        Flattened dictionary with deterministic key ordering.
    """

    flattened: dict[str, Any] = {}
    for key in sorted(values, key=str):
        value = values[key]
        flat_key = str(key) if prefix is None else f"{prefix}/{key}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def _to_python_array(matrix: Any) -> Any:
    """Convert common tensor-like matrices to Python or NumPy arrays.

    Args:
        matrix: Matrix-like object.

    Returns:
        Converted matrix accepted by matplotlib.
    """

    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    if hasattr(matrix, "numpy"):
        return matrix.numpy()
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return matrix


def _table_data(rows: Sequence[Mapping[str, Any]] | Sequence[Sequence[Any]]) -> tuple[list[str], list[list[Any]]]:
    """Convert row dictionaries or row sequences to W&B table inputs.

    Args:
        rows: Row dictionaries or row sequences.

    Returns:
        Column names and row data.
    """

    if not rows:
        return [], []

    first = rows[0]
    if isinstance(first, Mapping):
        string_rows = [{str(key): value for key, value in row.items()} for row in rows]  # type: ignore[union-attr]
        columns = sorted({key for row in string_rows for key in row})
        data = [[row.get(column) for column in columns] for row in string_rows]
        return columns, data

    width = max(len(row) for row in rows)  # type: ignore[arg-type]
    columns = [f"col_{index}" for index in range(width)]
    data = [list(row) + [None] * (width - len(row)) for row in rows]  # type: ignore[arg-type]
    return columns, data


__all__ = ["ExperimentLogger"]
