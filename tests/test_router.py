"""Unit tests for the auxiliary-loss-free Qwen3 MoE router."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alf.metrics import summarize_auxiliary_loss_free_router, summarize_expert_load
from alf.router import Qwen3MoeAuxiliaryLossFreeTopKRouter


def test_bias_changes_selection_but_not_selected_weights() -> None:
    """Bias should change selected experts while weights stay tied to raw probabilities."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        expert_bias_update_rate=0.0,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[4.0, 0.0], [3.0, 0.0], [1.0, 0.0]]))
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    hidden_states = torch.tensor([[1.0, 0.0]])
    router_logits, router_scores, router_indices = router(hidden_states)

    expected_probs = torch.softmax(router_logits, dim=-1)
    expected_scores = expected_probs.gather(dim=-1, index=router_indices)

    assert router_logits.shape == (1, 3)
    assert router_scores.shape == (1, 2)
    assert router_indices.shape == (1, 2)
    assert router_indices.tolist() == [[2, 0]]
    assert torch.allclose(router_scores, expected_scores)
    assert torch.equal(torch.argsort(expected_probs, dim=-1, descending=True)[:, :2], torch.tensor([[0, 1]]))


def test_norm_topk_prob_renormalizes_selected_weights() -> None:
    """Selected routing weights should be renormalized when requested."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        expert_bias_update_rate=0.0,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[4.0, 0.0], [3.0, 0.0], [1.0, 0.0]]))
        router.expert_bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    _, router_scores, router_indices = router(torch.tensor([[1.0, 0.0]]))
    raw_probs = torch.softmax(torch.tensor([[4.0, 3.0, 1.0]]), dim=-1)
    expected_scores = raw_probs.gather(dim=-1, index=router_indices)
    expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)

    assert torch.allclose(router_scores, expected_scores)
    assert torch.allclose(router_scores.sum(dim=-1), torch.ones(1))


def test_bias_updates_only_during_training_and_tracks_load_direction() -> None:
    """Overloaded experts should get lower bias while underloaded experts increase."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_init=0.0,
        expert_bias_update_rate=0.5,
        expert_bias_update_interval=1,
        expert_bias_warmup_steps=1,
        expert_bias_clip=0.2,
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    hidden_states = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    router.eval()
    router(hidden_states)
    assert torch.allclose(router.expert_bias, torch.tensor([0.0, 0.1]))
    assert int(router.training_steps.item()) == 0

    router.train()
    router(hidden_states)
    assert torch.allclose(router.expert_bias, torch.tensor([0.0, 0.1]))
    assert int(router.training_steps.item()) == 1
    assert int(router.bias_update_steps.item()) == 0

    router(hidden_states)
    assert torch.allclose(router.expert_bias, torch.tensor([0.2, -0.15]))
    assert router.expert_bias.requires_grad is False
    assert int(router.training_steps.item()) == 2
    assert int(router.bias_update_steps.item()) == 1
    assert router.last_expert_load.tolist() == [0, 4]


def test_sign_bias_update_policy_uses_fixed_step_direction() -> None:
    """Sign policy should use fixed-size updates from load direction only."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        expert_bias_update_policy="sign",
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    router.train()
    router(torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))

    assert torch.allclose(router.expert_bias, torch.tensor([0.1, 0.0]))
    assert torch.allclose(router.last_bias_delta, torch.tensor([0.1, -0.1]))


def test_router_metric_summary_is_serializable() -> None:
    """Router metric helpers should emit JSON-friendly values."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.0,
    )
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    router(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    load_summary = summarize_expert_load(counts=router.last_expert_load)
    router_summary = summarize_auxiliary_loss_free_router(router)

    assert load_summary["counts"] == [1, 1]
    assert load_summary["max_min_load_ratio"] == 1.0
    assert router_summary["bias"]["values"] == [0.0, 0.0]
    assert router_summary["load"]["total_assignments"] == 2
