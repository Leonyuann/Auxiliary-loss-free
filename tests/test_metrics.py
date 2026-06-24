"""Tests for metric helpers."""

import torch

from alf.metrics import load_balance_metrics


def test_load_balance_metrics() -> None:
    """Compute basic load-balancing statistics."""

    metrics = load_balance_metrics(torch.tensor([1, 2, 3]))

    assert metrics["expert_load_mean"] == 2.0
    assert metrics["expert_load_min"] == 1.0
    assert metrics["expert_load_max"] == 3.0
    assert metrics["expert_load_max_min_ratio"] == 3.0
