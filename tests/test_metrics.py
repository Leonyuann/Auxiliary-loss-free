"""Tests for metric helpers."""

import torch

from alf.metrics import (
    activation_matrix_from_counts,
    activation_rows_from_counts,
    add_bias_update_deltas,
    bias_update_matrix_from_deltas,
    bias_update_rows_from_deltas,
    collect_bias_update_deltas,
    collect_bias_update_steps,
    compute_maxvio,
    load_balance_metrics,
    mean_maxvio,
)
from alf.router import Qwen3MoeAuxiliaryLossFreeTopKRouter


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


def test_bias_update_delta_matrix_and_rows() -> None:
    """Collect newly applied router bias deltas as heatmap-ready data."""

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [
            Qwen3MoeAuxiliaryLossFreeTopKRouter(
                hidden_size=2,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                expert_bias_update_rate=0.1,
                expert_bias_update_policy="sign",
            ),
            Qwen3MoeAuxiliaryLossFreeTopKRouter(
                hidden_size=2,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                expert_bias_update_rate=0.2,
                expert_bias_update_policy="sign",
            ),
        ]
    )
    for router in model.layers:
        router.train()
        with torch.no_grad():
            router.weight.zero_()
            router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    previous_steps = collect_bias_update_steps(model)
    step_deltas: dict[str, torch.Tensor] = {}

    model.layers[0](torch.ones(4, 2))
    deltas, events = collect_bias_update_deltas(model, previous_steps)
    add_bias_update_deltas(step_deltas, deltas)

    assert events == 1
    assert set(deltas) == {"layers.0"}
    assert collect_bias_update_deltas(model, previous_steps) == ({}, 0)

    model.layers[1](torch.ones(4, 2))
    deltas, events = collect_bias_update_deltas(model, previous_steps)
    add_bias_update_deltas(step_deltas, deltas)

    matrix, layer_names = bias_update_matrix_from_deltas(step_deltas)
    rows = bias_update_rows_from_deltas(step_deltas, step=3)

    assert events == 1
    assert layer_names == ["layers.0", "layers.1"]
    assert torch.allclose(matrix, torch.tensor([[0.1, -0.1], [0.2, -0.2]]))
    assert rows[0] == {
        "step": 3,
        "layer_index": 0,
        "layer": "layers.0",
        "expert": 0,
        "bias_delta": 0.10000000149011612,
    }
