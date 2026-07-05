"""Tests for local C4 JSON.GZ token preparation helpers."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_prepare_module():
    """Load the C4 token preparation script as a test module."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_c4_bpe_tokens.py"
    spec = importlib.util.spec_from_file_location("prepare_c4_bpe_tokens", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchTokenizer:
    """Tokenizer test double that returns comma-separated integer ids."""

    eos_token_id = 0
    name_or_path = "test-tokenizer"

    def __call__(self, batch: list[str], add_special_tokens: bool = False) -> dict[str, list[list[int]]]:
        """Encode each text document as comma-separated integer ids."""

        del add_special_tokens
        return {"input_ids": [[int(value) for value in item.split(",") if value] for item in batch]}


def _write_jsonl_gz(path: Path, records: list[dict[str, str]]) -> None:
    """Write records to a gzipped JSONL file."""

    with gzip.open(path, "wt", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

def test_encode_c4_split_writes_tokens_and_metadata(tmp_path: Path) -> None:
    """Encode C4 JSON.GZ text fields into int32 tokens with metadata."""

    module = _load_prepare_module()
    c4_dir = tmp_path / "c4" / "en"
    c4_dir.mkdir(parents=True)
    shard = c4_dir / "c4-train.00000-of-00001.json.gz"
    _write_jsonl_gz(shard, [{"text": "1,2"}, {"text": "3"}, {"text": "4,5"}])

    shards = module.discover_shards(c4_dir, "c4-train.*.json.gz")
    output_path = tmp_path / "tokens.i32"
    metadata = module.encode_c4_split(
        tokenizer=BatchTokenizer(),
        shards=shards,
        output_path=output_path,
        max_tokens=5,
        batch_size=2,
        overwrite=False,
        split_name="train",
    )

    assert metadata["tokens"] == 5
    assert metadata["documents"] == 2
    assert metadata["num_shards"] == 1
    assert np.fromfile(output_path, dtype=np.int32).tolist() == [1, 2, 0, 3, 0]
    sidecar = json.loads(output_path.with_suffix(".i32.metadata.json").read_text(encoding="utf-8"))
    assert sidecar["tokenizer"] == "test-tokenizer"


def test_encode_c4_split_skips_existing_output_without_overwrite(tmp_path: Path) -> None:
    """Existing token files should be reused unless overwrite is requested."""

    module = _load_prepare_module()
    shard = tmp_path / "c4-validation.00000-of-00001.json.gz"
    _write_jsonl_gz(shard, [{"text": "9"}])
    output_path = tmp_path / "tokens.i32"
    np.asarray([7], dtype=np.int32).tofile(output_path)
    output_path.with_suffix(".i32.metadata.json").write_text(
        json.dumps({"output": str(output_path), "tokens": 1}),
        encoding="utf-8",
    )

    metadata = module.encode_c4_split(
        tokenizer=BatchTokenizer(),
        shards=[shard],
        output_path=output_path,
        max_tokens=4,
        batch_size=1,
        overwrite=False,
        split_name="validation",
    )

    assert metadata["tokens"] == 1
    assert np.fromfile(output_path, dtype=np.int32).tolist() == [7]
