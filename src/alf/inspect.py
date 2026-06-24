"""Inspect router metrics from an ALF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alf.metrics import collect_router_metrics
from alf.modeling import load_model_for_inspection


def inspect_router(checkpoint: str | Path) -> dict[str, object]:
    """Load a checkpoint and collect router metrics.

    Args:
        checkpoint: Model checkpoint directory.

    Returns:
        Router metrics dictionary.
    """

    model = load_model_for_inspection(Path(checkpoint))
    return collect_router_metrics(model)


def main() -> None:
    """Run the router inspection CLI."""

    parser = argparse.ArgumentParser(description="Inspect ALF router metrics.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory.")
    args = parser.parse_args()
    print(json.dumps(inspect_router(args.checkpoint), indent=2, sort_keys=True))
