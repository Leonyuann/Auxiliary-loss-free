"""Training observability helpers for ALF experiment logging."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import torch

from alf.metrics import compute_normalized_entropy, summarize_layerwise_normalized_entropy


class CudaStepTimer:
    """Measure full optimizer-step wall time with CUDA events when available.

    Attributes:
        device: Torch device used by the training step.
        cuda_enabled: Whether CUDA event timing is active.
        start_event: Optional CUDA start event.
        end_event: Optional CUDA end event.
        start_time: CPU fallback start timestamp.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize a step timer for one training device.

        Args:
            device: Device used for training.
        """

        self.device = device
        self.cuda_enabled = device.type == "cuda" and torch.cuda.is_available()
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None
        self.start_time = 0.0

    def start(self) -> None:
        """Start timing the current optimizer step."""

        if self.cuda_enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self.start_time = time.perf_counter()

    def stop_ms(self) -> float:
        """Stop timing and return elapsed milliseconds.

        Returns:
            Elapsed step time in milliseconds.
        """

        if self.cuda_enabled and self.start_event is not None and self.end_event is not None:
            self.end_event.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)
            return float(self.start_event.elapsed_time(self.end_event))
        return float((time.perf_counter() - self.start_time) * 1000.0)


class MovingAverage:
    """Fixed-window moving average for scalar metrics.

    Attributes:
        values: Recent scalar values.
    """

    def __init__(self, window_size: int = 100) -> None:
        """Create a moving-average window.

        Args:
            window_size: Maximum number of values retained.
        """

        self.values: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """Add a value and return the current average.

        Args:
            value: Scalar value to add.

        Returns:
            Mean of the retained window.
        """

        self.values.append(float(value))
        return float(sum(self.values) / max(len(self.values), 1))


class AllToAllProfiler:
    """Optional torch.profiler window for all-to-all communication metrics.

    Attributes:
        every_steps: Interval between profiling windows. Disabled when non-positive.
        window_steps: Number of consecutive steps profiled in each window.
        latest_metrics: Last measured profile metrics.
        _profiler: Active torch profiler instance, if any.
    """

    def __init__(self, *, every_steps: int = 0, window_steps: int = 0) -> None:
        """Initialize an all-to-all profiler.

        Args:
            every_steps: Start a profiling window every this many steps.
            window_steps: Number of steps per profiling window.
        """

        self.every_steps = int(every_steps)
        self.window_steps = int(window_steps)
        self.latest_metrics: dict[str, Any] | None = None
        self._profiler: Any | None = None

    @classmethod
    def from_env(cls) -> "AllToAllProfiler":
        """Create a profiler from environment variables.

        Returns:
            Profiler configured by ``ALF_PROFILE_ALL_TO_ALL_EVERY`` and
            ``ALF_PROFILE_ALL_TO_ALL_STEPS``. Profiling is disabled by default.
        """

        every_steps = int(os.environ.get("ALF_PROFILE_ALL_TO_ALL_EVERY", "0") or 0)
        window_steps = int(os.environ.get("ALF_PROFILE_ALL_TO_ALL_STEPS", "0") or 0)
        return cls(every_steps=every_steps, window_steps=window_steps)

    def enabled_for_step(self, step: int) -> bool:
        """Return whether a one-based step should be profiled.

        Args:
            step: One-based optimizer step.

        Returns:
            Whether profiling should be active for this step.
        """

        if self.every_steps <= 0 or self.window_steps <= 0:
            return False
        offset = (int(step) - 1) % self.every_steps
        return 0 <= offset < self.window_steps

    def start(self, step: int) -> None:
        """Start profiling a step when it is inside a configured window.

        Args:
            step: One-based optimizer step.
        """

        self.latest_metrics = None
        if not self.enabled_for_step(step):
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False, acc_events=True)
        self._profiler.__enter__()

    def stop(self, step_time_ms: float) -> dict[str, Any]:
        """Stop active profiling and return all-to-all metrics.

        Args:
            step_time_ms: Full optimizer-step time used to compute the ratio.

        Returns:
            Profile metrics. Empty when profiling was not active.
        """

        if self._profiler is None:
            return {}
        profiler = self._profiler
        self._profiler = None
        profiler.__exit__(None, None, None)
        all_to_all_us = 0.0
        for event in profiler.key_averages():
            key = str(getattr(event, "key", "")).lower()
            if "all_to_all" not in key and "alltoall" not in key:
                continue
            device_us = float(getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0) or 0.0)
            cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            all_to_all_us += max(device_us, cpu_us)
        all_to_all_ms = all_to_all_us / 1000.0
        ratio = all_to_all_ms / max(float(step_time_ms), 1e-9)
        self.latest_metrics = {
            "all_to_all_time_ms": float(all_to_all_ms),
            "all_to_all_time_ratio": float(ratio),
            "all_to_all_profile_active": True,
        }
        return dict(self.latest_metrics)


def gpu_memory_metrics(device: torch.device) -> dict[str, float]:
    """Return CUDA memory metrics for logging.

    Args:
        device: Training device.

    Returns:
        Dictionary with peak allocated and reserved memory in bytes. Returns zeros
        when CUDA is unavailable.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu_memory_allocated": 0.0, "gpu_memory_reserved": 0.0}
    return {
        "gpu_memory_allocated": float(torch.cuda.max_memory_allocated(device)),
        "gpu_memory_reserved": float(torch.cuda.max_memory_reserved(device)),
    }


def summarize_moe_observability(layer_counts: dict[str, torch.Tensor]) -> dict[str, float]:
    """Summarize MoE expert-load metrics for train logging.

    Args:
        layer_counts: Mapping from router names to per-expert assignment counts.

    Returns:
        Aggregate and layerwise MoE load metrics.
    """

    if not layer_counts:
        return {
            "expert_load_max_over_mean": 0.0,
            "expert_load_cv": 0.0,
            "expert_load_normalized_entropy": 0.0,
            "expert_load_layerwise_normalized_entropy_mean": 0.0,
            "expert_load_layerwise_normalized_entropy_min": 0.0,
            "expert_load_layerwise_normalized_entropy_max": 0.0,
            "overflow_rate": 0.0,
            "dropped_token_rate": 0.0,
        }
    aggregate = torch.stack([counts.detach().to(dtype=torch.float32, device="cpu") for counts in layer_counts.values()]).sum(dim=0)
    mean_load = float(aggregate.mean().item()) if aggregate.numel() else 0.0
    max_over_mean = float(aggregate.max().item() / mean_load) if mean_load > 0.0 and aggregate.numel() else 0.0
    cv = float(aggregate.std(unbiased=False).item() / mean_load) if mean_load > 0.0 and aggregate.numel() else 0.0
    entropy_summary = summarize_layerwise_normalized_entropy(layer_counts)
    return {
        "expert_load_max_over_mean": max_over_mean,
        "expert_load_cv": cv,
        "expert_load_normalized_entropy": compute_normalized_entropy(aggregate),
        "expert_load_layerwise_normalized_entropy_mean": entropy_summary["mean"],
        "expert_load_layerwise_normalized_entropy_min": entropy_summary["min"],
        "expert_load_layerwise_normalized_entropy_max": entropy_summary["max"],
        "overflow_rate": 0.0,
        "dropped_token_rate": 0.0,
    }


__all__ = [
    "AllToAllProfiler",
    "CudaStepTimer",
    "MovingAverage",
    "gpu_memory_metrics",
    "summarize_moe_observability",
]
