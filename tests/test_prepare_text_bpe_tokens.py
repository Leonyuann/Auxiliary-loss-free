"""Tests for OWT BPE token preparation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_prepare_module():
    """Load the token preparation script as a test module."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_text_bpe_tokens.py"
    spec = importlib.util.spec_from_file_location("prepare_text_bpe_tokens", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchTokenizer:
    """Tokenizer test double that returns configured token ids per document."""

    eos_token_id = 0

    def __call__(self, batch: list[str], add_special_tokens: bool = False) -> dict[str, list[list[int]]]:
        """Encode each text document as comma-separated integer ids."""

        del add_special_tokens
        return {"input_ids": [[int(value) for value in item.split(",") if value] for item in batch]}


class ProgressRecorder:
    """Progress-bar test double that records update calls."""

    def __init__(self) -> None:
        """Initialize an empty progress update log."""

        self.updates: list[int] = []

    def update(self, value: int) -> None:
        """Record one progress update value."""

        self.updates.append(value)


def test_flush_batch_writes_contiguous_tokens_and_truncates(tmp_path) -> None:
    """Flush one encoded batch with EOS tokens and max-token truncation."""

    module = _load_prepare_module()
    output_path = tmp_path / "tokens.i32"
    progress = ProgressRecorder()

    with output_path.open("wb") as file:
        written, docs = module._flush_batch(
            BatchTokenizer(),
            ["1,2", "3", "4,5,6"],
            file,
            max_tokens=6,
            written=0,
            docs=0,
            progress=progress,
        )

    assert written == 6
    assert docs == 3
    assert progress.updates == [6]
    assert np.fromfile(output_path, dtype=np.int32).tolist() == [1, 2, 0, 3, 0, 4]
