"""Unit tests for the auxiliary-loss-free Qwen3 MoE router."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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


def test_bias_updates_only_during_training_and_tracks_load_direction_without_default_clip() -> None:
    """Overloaded experts should get lower bias without default clipping."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_init=0.0,
        expert_bias_update_rate=0.5,
        expert_bias_update_interval=1,
        expert_bias_warmup_steps=1,
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
    assert int(router.training_steps.item()) == 0
    assert int(router.bias_update_steps.item()) == 0

    assert router.update_expert_bias_from_accumulated_load() is False
    assert torch.allclose(router.expert_bias, torch.tensor([0.0, 0.1]))
    assert int(router.training_steps.item()) == 1
    assert int(router.bias_update_steps.item()) == 0

    router(hidden_states)
    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.allclose(router.expert_bias, torch.tensor([0.25, -0.15]))
    assert router.expert_bias.requires_grad is False
    assert int(router.training_steps.item()) == 2
    assert int(router.bias_update_steps.item()) == 1
    assert router.last_expert_load.tolist() == [0, 4]


def test_bias_stops_updating_after_max_optimizer_step() -> None:
    """Bias should freeze after the configured optimizer-step boundary."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        expert_bias_update_policy="sign",
        expert_bias_max_update_steps=1,
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    router.train()
    hidden_states = torch.ones(4, 2)
    router(hidden_states)
    assert router.update_expert_bias_from_accumulated_load() is True
    frozen_bias = router.expert_bias.detach().clone()

    router(hidden_states)
    assert router.update_expert_bias_from_accumulated_load() is False
    assert torch.equal(router.expert_bias, frozen_bias)
    assert torch.equal(router.last_bias_delta, torch.zeros(2))
    assert int(router.training_steps.item()) == 2
    assert int(router.bias_update_steps.item()) == 1


def test_negative_bias_max_update_steps_raises() -> None:
    """The bias update boundary must be non-negative when configured."""

    try:
        Qwen3MoeAuxiliaryLossFreeTopKRouter(
            hidden_size=2,
            num_experts=2,
            num_experts_per_tok=1,
            norm_topk_prob=False,
            expert_bias_max_update_steps=-1,
        )
    except ValueError as error:
        assert "expert_bias_max_update_steps" in str(error)
    else:
        raise AssertionError("Expected ValueError")


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

    assert torch.allclose(router.expert_bias, torch.tensor([0.0, 0.1]))
    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.allclose(router.expert_bias, torch.tensor([0.1, 0.0]))
    assert torch.allclose(router.last_bias_delta, torch.tensor([0.1, -0.1]))


def test_bias_update_uses_accumulated_microbatch_load_once_per_step() -> None:
    """Bias updates should use accumulated load across micro batches."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        expert_bias_update_policy="sign",
    )
    with torch.no_grad():
        router.weight.copy_(torch.eye(2))

    router.train()
    router(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    router(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    assert torch.allclose(router.expert_bias, torch.zeros(2))
    assert router.accumulated_expert_load.tolist() == [2, 2]
    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.allclose(router.last_bias_delta, torch.zeros(2))
    assert torch.allclose(router.expert_bias, torch.zeros(2))
    assert int(router.bias_update_steps.item()) == 1
    assert router.accumulated_expert_load.tolist() == [0, 0]


def test_balanced_topk_sign_updates_equal_positive_and_negative_extremes() -> None:
    """Balanced top-k sign updates only the largest errors on each side."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=6,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        expert_bias_update_policy="balanced_topk_sign",
        expert_bias_update_topk=2,
    )

    target_fraction = torch.full((6,), 1.0 / 6.0)
    load_error = torch.tensor([0.01, 0.02, 0.03, -0.01, -0.02, -0.03])
    with torch.no_grad():
        router.last_load_fraction.copy_(target_fraction - load_error)

    router._update_expert_bias()

    assert torch.allclose(router.last_bias_delta, torch.tensor([0.0, 0.1, 0.1, 0.0, -0.1, -0.1]))
    assert torch.allclose(router.expert_bias, torch.tensor([0.0, 0.1, 0.1, 0.0, -0.1, -0.1]))
    assert int(router.bias_update_steps.item()) == 1


def test_control_buffers_stay_float32_after_bfloat16_cast() -> None:
    """Small bias updates should survive when model weights use bfloat16."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=1e-3,
        expert_bias_update_policy="sign",
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.5, 0.6]))

    router.to(dtype=torch.bfloat16)

    assert router.weight.dtype == torch.bfloat16
    assert router.expert_bias.dtype == torch.float32
    assert router.last_load_fraction.dtype == torch.float32
    assert router.last_bias_delta.dtype == torch.float32
    assert router.last_normalized_feedback_gain.dtype == torch.float32
    assert router.load_error_ema.dtype == torch.float32
    assert router.load_error_accumulator.dtype == torch.float32
    assert router.previous_load_error.dtype == torch.float32
    assert router.persistent_energy_ema.dtype == torch.float32
    assert router.oscillation_energy_ema.dtype == torch.float32
    assert router.last_adaptive_ema_beta.dtype == torch.float32

    router.train()
    router(torch.ones(4, 2, dtype=torch.bfloat16))
    assert torch.allclose(router.expert_bias, torch.tensor([0.5, 0.6]))

    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.allclose(router.last_bias_delta, torch.tensor([1e-3, -1e-3]))
    assert torch.allclose(router.expert_bias, torch.tensor([0.501, 0.599]))


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
    assert abs(router_summary["adaptive_ema_beta"] - 0.95) < 1e-6
    assert router_summary["gain_coupled_normalized_gain"] == pytest.approx(1.0 / 30.0)
    assert router_summary["gain_coupled_rate_min"] == 0.05
    assert router_summary["gain_coupled_rate_max"] == 0.3
    assert router_summary["normalized_feedback_gain"] == 0.0
    assert router_summary["excess_load_variance"] == 0.0
    assert router_summary["persistent_energy_ema"] == 0.0
    assert router_summary["oscillation_energy_ema"] == 0.0


def test_ema_bias_update_policy_tracks_smoothed_error() -> None:
    """EMA policy should smooth load error before applying bias updates."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=1.0,
        expert_bias_update_policy="ema",
        expert_bias_ema_beta=0.5,
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    router.train()
    router(torch.ones(4, 2))

    assert torch.allclose(router.load_error_ema, torch.zeros(2))
    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.allclose(router.load_error_ema, torch.tensor([0.25, -0.25]))
    assert torch.allclose(router.last_bias_delta, torch.tensor([0.25, -0.25]))


def test_adaptive_ema_variance_uses_noise_corrected_load_variance() -> None:
    """Variance-adaptive EMA should lower beta for excess load imbalance."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=1.0,
        expert_bias_update_policy="adaptive_ema_variance",
        expert_bias_adaptive_beta_min=0.1,
        expert_bias_adaptive_beta_max=0.9,
        expert_bias_adaptive_variance_reference=0.25,
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    router.train()
    router(torch.ones(4, 2))

    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.isclose(router.last_normalized_load_variance, torch.tensor(1.0))
    assert torch.isclose(router.last_batch_noise, torch.tensor(0.25))
    assert torch.isclose(router.last_excess_load_variance, torch.tensor(0.75))
    assert torch.isclose(router.last_adaptive_ema_beta, torch.tensor(0.3))
    assert torch.allclose(router.load_error_ema, torch.tensor([0.35, -0.35]))
    assert torch.allclose(router.last_bias_delta, router.load_error_ema)


def test_adaptive_ema_persistent_oscillation_raises_beta_on_reversal() -> None:
    """Persistent/oscillation EMA should smooth a reversing load-error direction."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=1.0,
        expert_bias_update_policy="adaptive_ema_persistent_oscillation",
        expert_bias_adaptive_beta_min=0.1,
        expert_bias_adaptive_beta_max=0.9,
        expert_bias_adaptive_state_decay=0.0,
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    router.train()
    router(torch.ones(4, 2))
    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.isclose(router.persistent_energy_ema, torch.tensor(0.875))
    assert torch.isclose(router.oscillation_energy_ema, torch.tensor(0.0))
    assert torch.isclose(router.last_adaptive_ema_beta, torch.tensor(2.0 / 9.0))
    assert torch.allclose(
        router.load_error_ema,
        torch.tensor([7.0 / 18.0, -7.0 / 18.0]),
    )

    with torch.no_grad():
        router.last_load_fraction.copy_(torch.tensor([1.0, 0.0]))
    router._update_expert_bias()

    assert torch.isclose(router.persistent_energy_ema, torch.tensor(0.0))
    assert torch.isclose(router.oscillation_energy_ema, torch.tensor(0.875))
    assert torch.isclose(router.last_adaptive_ema_beta, torch.tensor(0.9))
    assert torch.allclose(router.load_error_ema, torch.tensor([0.3, -0.3]))


def test_persistent_oscillation_energies_remove_split_noise_before_slow_ema() -> None:
    """Persistent and oscillation energies should denoise before slow EMA."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        expert_bias_update_policy="adaptive_ema_persistent_oscillation",
        expert_bias_adaptive_beta_min=0.0,
        expert_bias_adaptive_beta_max=0.99,
        expert_bias_adaptive_state_decay=0.5,
    )

    with torch.no_grad():
        router.last_expert_load.copy_(torch.tensor([60, 40]))
        router.last_load_fraction.copy_(torch.tensor([0.6, 0.4]))
    router._update_expert_bias()

    assert router.last_batch_noise.item() == pytest.approx(0.01)
    assert router.persistent_energy_ema.item() == pytest.approx(0.035)
    assert router.oscillation_energy_ema.item() == pytest.approx(0.0)

    with torch.no_grad():
        router.last_expert_load.copy_(torch.tensor([45, 55]))
        router.last_load_fraction.copy_(torch.tensor([0.45, 0.55]))
    router._update_expert_bias()

    assert router.last_batch_noise.item() == pytest.approx(0.01)
    assert router.persistent_energy_ema.item() == pytest.approx(0.0175)
    assert router.oscillation_energy_ema.item() == pytest.approx(0.00875)


def test_gain_coupled_adaptive_ema_reuses_persistent_beta_estimator() -> None:
    """Gain coupling should change only rate while preserving adaptive EMA state."""

    common_kwargs = {
        "hidden_size": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "norm_topk_prob": False,
        "expert_bias_update_rate": 0.1,
        "expert_bias_adaptive_beta_min": 0.25,
        "expert_bias_adaptive_beta_max": 0.75,
        "expert_bias_adaptive_state_decay": 0.0,
    }
    fixed_rate = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        **common_kwargs,
        expert_bias_update_policy="adaptive_ema_persistent_oscillation",
    )
    gain_coupled = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        **common_kwargs,
        expert_bias_update_policy="adaptive_ema_gain_coupled",
        expert_bias_gain_coupled_normalized_gain=1.0 / 30.0,
        expert_bias_gain_coupled_rate_min=0.05,
        expert_bias_gain_coupled_rate_max=0.3,
    )

    expected_betas = [0.25, 0.75]
    expected_rates = [1.0 / 18.0, 7.0 / 30.0]
    for load_fraction, expected_beta, expected_rate in zip(
        (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])),
        expected_betas,
        expected_rates,
        strict=True,
    ):
        for router in (fixed_rate, gain_coupled):
            with torch.no_grad():
                router.last_expert_load.copy_((load_fraction * 4).to(dtype=torch.long))
                router.last_load_fraction.copy_(load_fraction)
            router._update_expert_bias()

        assert torch.equal(gain_coupled.last_adaptive_ema_beta, fixed_rate.last_adaptive_ema_beta)
        assert torch.equal(gain_coupled.load_error_ema, fixed_rate.load_error_ema)
        assert gain_coupled.last_adaptive_ema_beta.item() == pytest.approx(expected_beta)
        assert fixed_rate.last_bias_update_rate.item() == pytest.approx(0.1)
        assert gain_coupled.last_bias_update_rate.item() == pytest.approx(expected_rate)
        assert gain_coupled.last_normalized_feedback_gain.item() == pytest.approx(1.0 / 30.0)


def test_gain_coupled_adaptive_ema_clips_dynamic_rate() -> None:
    """Gain-coupled updates should respect configured dynamic-rate bounds."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_policy="adaptive_ema_gain_coupled",
        expert_bias_adaptive_beta_min=0.9,
        expert_bias_adaptive_beta_max=0.9,
        expert_bias_gain_coupled_normalized_gain=1.0 / 30.0,
        expert_bias_gain_coupled_rate_min=0.05,
        expert_bias_gain_coupled_rate_max=0.3,
    )
    with torch.no_grad():
        router.last_expert_load.copy_(torch.tensor([0, 4]))
        router.last_load_fraction.copy_(torch.tensor([0.0, 1.0]))
    router._update_expert_bias()

    assert router.last_adaptive_ema_beta.item() == pytest.approx(0.9)
    assert router.last_bias_update_rate.item() == pytest.approx(0.3)
    assert router.last_normalized_feedback_gain.item() == pytest.approx(0.3 / 19.0)


def test_adaptive_ema_state_round_trips_through_router_state_dict() -> None:
    """Adaptive beta history should survive model checkpoint save and restore."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=1.0,
        expert_bias_update_policy="adaptive_ema_persistent_oscillation",
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))
    router.train()
    router(torch.ones(4, 2))
    router.update_expert_bias_from_accumulated_load()

    restored = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=1.0,
        expert_bias_update_policy="adaptive_ema_persistent_oscillation",
    )
    restored.load_state_dict(router.state_dict())

    assert bool(restored.adaptive_state_initialized.item()) is True
    assert torch.equal(restored.previous_load_error, router.previous_load_error)
    assert torch.equal(restored.persistent_energy_ema, router.persistent_energy_ema)
    assert torch.equal(restored.oscillation_energy_ema, router.oscillation_energy_ema)
    assert torch.equal(restored.last_normalized_feedback_gain, router.last_normalized_feedback_gain)
    assert torch.equal(restored.last_adaptive_ema_beta, router.last_adaptive_ema_beta)


def test_accumulated_sign_updates_only_on_interval() -> None:
    """Accumulated sign policy should delay bias writes until the interval boundary."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.1,
        expert_bias_update_policy="accumulated_sign",
        expert_bias_update_interval=2,
    )
    with torch.no_grad():
        router.weight.zero_()
        router.expert_bias.copy_(torch.tensor([0.0, 0.1]))

    router.train()
    router(torch.ones(4, 2))
    assert router.update_expert_bias_from_accumulated_load() is False
    assert torch.allclose(router.last_bias_delta, torch.zeros(2))
    assert int(router.bias_update_steps.item()) == 0
    assert torch.allclose(router.load_error_accumulator, torch.tensor([0.5, -0.5]))

    router(torch.ones(4, 2))
    assert router.update_expert_bias_from_accumulated_load() is True
    assert torch.allclose(router.last_bias_delta, torch.tensor([0.1, -0.1]))
    assert int(router.bias_update_steps.item()) == 1
    assert torch.allclose(router.load_error_accumulator, torch.zeros(2))


def test_linear_bias_update_schedule_decays_to_end_rate() -> None:
    """Linear bias schedule should decay update magnitude over post-warmup steps."""

    router = Qwen3MoeAuxiliaryLossFreeTopKRouter(
        hidden_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
        expert_bias_update_rate=0.3,
        expert_bias_update_policy="sign",
        expert_bias_update_schedule="linear",
        expert_bias_update_schedule_steps=3,
        expert_bias_update_end_rate=0.0,
    )
    observed_fraction = torch.tensor([0.0, 1.0])

    expected_rates = [0.3, 0.15, 0.0, 0.0]
    for expected_rate in expected_rates:
        with torch.no_grad():
            router.last_load_fraction.copy_(observed_fraction)
        router._update_expert_bias()
        assert torch.allclose(router.last_bias_delta, torch.tensor([expected_rate, -expected_rate]))
        assert torch.isclose(router.last_bias_update_rate, torch.tensor(expected_rate))

    assert torch.allclose(router.expert_bias, torch.tensor([0.45, -0.45]))


def test_invalid_linear_bias_update_schedule_raises() -> None:
    """Linear schedule requires an explicit positive schedule length."""

    for schedule_steps in (None, 0):
        try:
            Qwen3MoeAuxiliaryLossFreeTopKRouter(
                hidden_size=2,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                expert_bias_update_schedule="linear",
                expert_bias_update_schedule_steps=schedule_steps,
            )
        except ValueError as error:
            assert "expert_bias_update_schedule_steps" in str(error)
        else:
            raise AssertionError("Expected ValueError")


def test_invalid_adaptive_ema_parameters_raise() -> None:
    """Adaptive EMA bounds, variance reference, and state decay must be valid."""

    invalid_kwargs = [
        {"expert_bias_adaptive_beta_min": -0.1},
        {"expert_bias_adaptive_beta_min": 0.9, "expert_bias_adaptive_beta_max": 0.8},
        {"expert_bias_adaptive_beta_max": 1.0},
        {"expert_bias_adaptive_variance_reference": 0.0},
        {"expert_bias_adaptive_state_decay": 1.0},
    ]
    for kwargs in invalid_kwargs:
        try:
            Qwen3MoeAuxiliaryLossFreeTopKRouter(
                hidden_size=2,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                **kwargs,
            )
        except ValueError as error:
            assert "adaptive" in str(error)
        else:
            raise AssertionError("Expected ValueError")


def test_invalid_gain_coupled_parameters_raise() -> None:
    """Gain-coupled normalized gain, rate bounds, and schedules must be valid."""

    invalid_kwargs = [
        {"expert_bias_gain_coupled_normalized_gain": -0.1},
        {"expert_bias_gain_coupled_rate_min": -0.1},
        {"expert_bias_gain_coupled_rate_min": 0.4, "expert_bias_gain_coupled_rate_max": 0.3},
        {
            "expert_bias_update_policy": "adaptive_ema_gain_coupled",
            "expert_bias_update_schedule": "linear",
            "expert_bias_update_schedule_steps": 2,
        },
    ]
    for kwargs in invalid_kwargs:
        try:
            Qwen3MoeAuxiliaryLossFreeTopKRouter(
                hidden_size=2,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                **kwargs,
            )
        except ValueError as error:
            assert "gain" in str(error)
        else:
            raise AssertionError("Expected ValueError")


def test_invalid_ema_beta_raises() -> None:
    """EMA beta must be in the half-open interval [0, 1)."""

    try:
        Qwen3MoeAuxiliaryLossFreeTopKRouter(
            hidden_size=2,
            num_experts=2,
            num_experts_per_tok=1,
            norm_topk_prob=False,
            expert_bias_update_policy="ema",
            expert_bias_ema_beta=1.0,
        )
    except ValueError as error:
        assert "expert_bias_ema_beta" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_invalid_balanced_topk_raises() -> None:
    """Balanced top-k count must be positive and fit the expert count."""

    for topk in (0, 3):
        try:
            Qwen3MoeAuxiliaryLossFreeTopKRouter(
                hidden_size=2,
                num_experts=2,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                expert_bias_update_policy="balanced_topk_sign",
                expert_bias_update_topk=topk,
            )
        except ValueError as error:
            assert "expert_bias_update_topk" in str(error)
        else:
            raise AssertionError("Expected ValueError")
