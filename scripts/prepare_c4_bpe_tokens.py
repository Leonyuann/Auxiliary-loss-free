#!/usr/bin/env python3
"""Encode local C4 JSON.GZ shards into contiguous int32 BPE token files."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DEFAULT_C4_DIR = Path("/vepfs-mlp2/ylq/data/c4/en")
DEFAULT_TOKENIZER_DIR = Path("/vepfs-mlp2/ylq/tokenizers/owt_bpe_32k")
DEFAULT_TRAIN_OUTPUT = Path("/vepfs-mlp2/ylq/data/c4/c4_train_owt_bpe32k_tokens.i32")
DEFAULT_VALIDATION_OUTPUT = Path("/vepfs-mlp2/ylq/data/c4/c4_validation_owt_bpe32k_tokens.i32")
DEFAULT_TRAIN_TOKENS = 10_000_000_000
DEFAULT_VALIDATION_TOKENS = 16_777_216


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments for C4 token preparation.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c4-dir", type=Path, default=DEFAULT_C4_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--train-pattern", default="c4-train.*.json.gz")
    parser.add_argument("--validation-pattern", default="c4-validation.*.json.gz")
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=DEFAULT_TRAIN_TOKENS,
        help="Number of new train tokens to append in this invocation.",
    )
    parser.add_argument(
        "--max-validation-tokens",
        type=int,
        default=DEFAULT_VALIDATION_TOKENS,
        help="Number of new validation tokens to append in this invocation.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run C4 token preparation for train and validation splits."""

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    train_shards = discover_shards(args.c4_dir, args.train_pattern)
    validation_shards = discover_shards(args.c4_dir, args.validation_pattern)
    encode_c4_split(
        tokenizer=tokenizer,
        shards=train_shards,
        output_path=args.train_output,
        max_tokens=args.max_train_tokens,
        batch_size=args.encode_batch_size,
        overwrite=args.overwrite,
        split_name="train",
    )
    encode_c4_split(
        tokenizer=tokenizer,
        shards=validation_shards,
        output_path=args.validation_output,
        max_tokens=args.max_validation_tokens,
        batch_size=args.encode_batch_size,
        overwrite=args.overwrite,
        split_name="validation",
    )


def discover_shards(c4_dir: Path, pattern: str) -> list[Path]:
    """Return sorted C4 shard paths for a split pattern.

    Args:
        c4_dir: Directory containing local C4 shard files.
        pattern: Glob pattern such as ``c4-train.*.json.gz``.

    Returns:
        Sorted shard paths.

    Raises:
        FileNotFoundError: If the directory or matching shards are missing.
    """

    if not c4_dir.exists():
        raise FileNotFoundError(f"C4 directory not found: {c4_dir}")
    shards = sorted(c4_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No C4 shards matched {pattern!r} under {c4_dir}")
    return shards


def iter_c4_texts(shards: Sequence[Path]) -> Iterable[str]:
    """Yield text fields from gzipped C4 JSONL shards.

    Args:
        shards: C4 ``.json.gz`` shard paths.

    Yields:
        Non-empty text fields from the shards.

    Raises:
        ValueError: If a line is not valid JSON or lacks a string ``text`` field.
    """

    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8", errors="ignore") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {shard}:{line_number}: {error}") from error
                text = record.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Missing string text field in {shard}:{line_number}")
                if text:
                    yield text


def encode_c4_split(
    *,
    tokenizer: Any,
    shards: Sequence[Path],
    output_path: Path,
    max_tokens: int,
    batch_size: int,
    overwrite: bool,
    split_name: str,
) -> dict[str, Any]:
    """Encode or append one C4 split into a contiguous int32 token file.

    Args:
        tokenizer: Batch tokenizer with a Hugging Face-compatible call API.
        shards: Ordered C4 shard paths to encode.
        output_path: Destination ``.i32`` token file.
        max_tokens: Maximum number of new tokens to write in this invocation.
        batch_size: Number of documents per tokenizer call.
        overwrite: Whether existing outputs should be rebuilt from the beginning.
        split_name: Human-readable split name for progress and metadata.

    Returns:
        Metadata dictionary written beside the output file.

    Raises:
        ValueError: If numeric limits are invalid, or an existing token file lacks
            the document-count metadata needed to append safely.
        FileExistsError: If a temporary rebuild output exists without overwrite.
    """

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_metadata = None if overwrite or not output_path.exists() else _read_existing_metadata(output_path)
    previous_tokens = 0
    previous_docs = 0
    if existing_metadata is not None:
        if "documents" not in existing_metadata:
            raise ValueError(
                "Existing C4 token file cannot be appended because its metadata "
                f"does not contain a document count: {_metadata_path(output_path)}. "
                "Rebuild once with --overwrite."
            )
        previous_tokens = int(existing_metadata.get("tokens", 0))
        expected_size = previous_tokens * np.dtype(np.int32).itemsize
        actual_size = output_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "Existing C4 token file size does not match metadata: "
                f"size={actual_size}, expected={expected_size}. Rebuild with --overwrite."
            )
        previous_docs = int(existing_metadata["documents"])
        destination_path = output_path
        open_mode = "ab"
        desc = f"Appending C4 {split_name}"
    else:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if tmp_path.exists():
            if overwrite:
                tmp_path.unlink()
            else:
                raise FileExistsError(f"Temporary output already exists: {tmp_path}")
        destination_path = tmp_path
        open_mode = "wb"
        desc = f"Encoding C4 {split_name}"

    written = 0
    docs = 0
    batch: list[str] = []
    with destination_path.open(open_mode) as destination:
        progress = tqdm(
            total=max_tokens,
            desc=desc,
            unit="tok",
            disable=not sys.stderr.isatty(),
        )
        try:
            for document_index, text in enumerate(iter_c4_texts(shards)):
                if document_index < previous_docs:
                    continue
                if written >= max_tokens:
                    break
                batch.append(text)
                if len(batch) >= batch_size:
                    written, docs = _flush_text_batch(tokenizer, batch, destination, max_tokens, written, docs, progress)
                    batch.clear()
            if batch and written < max_tokens:
                written, docs = _flush_text_batch(tokenizer, batch, destination, max_tokens, written, docs, progress)
        finally:
            progress.close()

    if existing_metadata is None:
        destination_path.replace(output_path)
        runs: list[dict[str, Any]] = []
    else:
        runs = list(existing_metadata.get("runs", []))

    run_record = {
        "mode": "overwrite" if overwrite else ("append" if existing_metadata is not None else "create"),
        "requested_tokens": max_tokens,
        "tokens": written,
        "documents": docs,
        "start_tokens": previous_tokens,
        "start_documents": previous_docs,
        "end_tokens": previous_tokens + written,
        "end_documents": previous_docs + docs,
    }
    runs.append(run_record)
    metadata = {
        "split": split_name,
        "output": str(output_path),
        "tokens": previous_tokens + written,
        "documents": previous_docs + docs,
        "last_run_tokens": written,
        "last_run_documents": docs,
        "last_requested_tokens": max_tokens,
        "dtype": "int32",
        "batch_size": batch_size,
        "shards": [str(path) for path in shards],
        "num_shards": len(shards),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "runs": runs,
    }
    _metadata_path(output_path).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    action = "Appended" if existing_metadata is not None else "Wrote"
    print(f"{action} {written} new tokens from {docs} docs to {output_path}")
    return metadata


def _metadata_path(output_path: Path) -> Path:
    """Return the sidecar metadata path for a token file.

    Args:
        output_path: Token file path.

    Returns:
        Sidecar metadata path.
    """

    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _read_existing_metadata(output_path: Path) -> dict[str, Any]:
    """Read existing metadata for an appendable token file.

    Args:
        output_path: Token file whose sidecar metadata should be loaded.

    Returns:
        Metadata dictionary.

    Raises:
        ValueError: If the sidecar metadata is missing.
    """

    metadata_path = _metadata_path(output_path)
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    raise ValueError(f"Existing C4 token file is missing sidecar metadata: {metadata_path}")


def _flush_text_batch(
    tokenizer: Any,
    batch: list[str],
    destination: Any,
    max_tokens: int,
    written: int,
    docs: int,
    progress: Any,
) -> tuple[int, int]:
    """Encode and write one batch of C4 documents.

    Args:
        tokenizer: Tokenizer used to encode text documents.
        batch: Text documents to encode.
        destination: Binary output file handle.
        max_tokens: Maximum number of tokens to write across the split.
        written: Number of tokens already written.
        docs: Number of documents already consumed.
        progress: Progress bar updated after writing.

    Returns:
        Updated ``(written, docs)`` counters.
    """

    encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_id = None if eos is None else int(eos)
    remaining = max_tokens - written
    if remaining <= 0:
        return written, docs

    output_ids: list[int] = []
    batch_docs = 0
    for ids in encoded:
        if remaining <= 0:
            break
        ids_len = len(ids)
        document_tokens = ids_len + (1 if eos_id is not None else 0)
        if document_tokens > remaining:
            if remaining <= ids_len:
                output_ids.extend(ids[:remaining])
            else:
                output_ids.extend(ids)
                output_ids.append(eos_id)
            remaining = 0
        else:
            output_ids.extend(ids)
            if eos_id is not None:
                output_ids.append(eos_id)
            remaining -= document_tokens
        batch_docs += 1

    if not output_ids:
        return written, docs

    array = np.asarray(output_ids, dtype=np.int32)
    array.tofile(destination)
    written += int(array.size)
    docs += batch_docs
    progress.update(int(array.size))
    return written, docs


if __name__ == "__main__":
    main()
