"""Tests for metric helpers."""

import torch

from alf.metrics import activation_matrix_from_counts, activation_rows_from_counts, compute_maxvio, load_balance_metrics, mean_maxvio


def test_load_balance_metrics() -> None:
    """Compute basic load-balancing statistics."""

    metrics = load_balance_metrics(torch.tensor([1, 2, 3]))

    assert metrics["expert_load_mean"] == 2.0
    assert metrics["expert_load_min"] == 1.0
    assert metrics["expert_load_max"] == 3.0
    assert metrics["expert_load_max_min_ratio"] == 3.0


def test_compute_maxvio_for_balanced_and_unbalanced_counts() -> None:
    """Compute maximal violation from per-expert counts."""

    assert compute_maxvio(torch.tensor([4, 4, 4, 4])) == 0.0
    assert compute_maxvio(torch.tensor([8, 4, 4, 0])) == 1.0
    assert compute_maxvio(torch.tensor([0, 0, 0, 0])) == 0.0


def test_activation_matrix_and_rows_from_counts() -> None:
    """Convert layer counts into heatmap matrix and table rows."""

    counts = {
        "model.layers.10.mlp.gate": torch.tensor([1, 3]),
        "model.layers.2.mlp.gate": torch.tensor([3, 1]),
        "model.layers.0.mlp.gate": torch.tensor([2, 2]),
    }

    matrix, layer_names = activation_matrix_from_counts(counts)
    rows = activation_rows_from_counts(counts, step=7, split="train")

    assert layer_names == ["model.layers.0.mlp.gate", "model.layers.2.mlp.gate", "model.layers.10.mlp.gate"]
    assert torch.allclose(matrix, torch.tensor([[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]))
    assert mean_maxvio(counts) == 1 / 3
    assert rows[0] == {
        "step": 7,
        "split": "train",
        "layer_index": 0,
        "layer": "model.layers.0.mlp.gate",
        "expert": 0,
        "count": 2,
        "fraction": 0.5,
    }
