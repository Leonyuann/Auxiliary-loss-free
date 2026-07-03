#!/usr/bin/env python3
"""Train/reuse a byte-level BPE tokenizer and encode local text into int32 tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--train-tokenizer-input", type=Path)
    parser.add_argument("--tokenizer-train-max-docs", type=int, default=200_000)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--encode-batch-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-train-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Token file already exists, skipping: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ensure_tokenizer(args)
    encode_text_file(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        max_tokens=args.max_tokens,
        batch_size=args.encode_batch_size,
        overwrite=args.overwrite,
    )


def ensure_tokenizer(args: argparse.Namespace):
    tokenizer_json = args.tokenizer_dir / "tokenizer.json"
    if tokenizer_json.exists() and not args.force_train_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        if len(tokenizer) != args.vocab_size:
            raise ValueError(f"Tokenizer vocab mismatch: len={len(tokenizer)}, expected={args.vocab_size}")
        return tokenizer

    train_input = args.train_tokenizer_input or args.input
    if not train_input.exists():
        raise FileNotFoundError(f"Tokenizer training input not found: {train_input}")
    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.tokenizer_dir / "tokenizer_train_sample.txt"
    with train_input.open("r", encoding="utf-8", errors="ignore") as src, sample_path.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if index >= args.tokenizer_train_max_docs:
                break
            dst.write(line)

    raw = ByteLevelBPETokenizer()
    raw.train(
        files=[str(sample_path)],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=raw._tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast.save_pretrained(args.tokenizer_dir)
    metadata = {
        "type": "byte-level-bpe",
        "vocab_size": len(fast),
        "special_tokens": SPECIAL_TOKENS,
        "train_input": str(train_input),
        "train_max_docs": args.tokenizer_train_max_docs,
    }
    (args.tokenizer_dir / "alf_tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if len(fast) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab mismatch after training: len={len(fast)}, expected={args.vocab_size}")
    return fast


def encode_text_file(*, tokenizer, input_path: Path, output_path: Path, max_tokens: int, batch_size: int, overwrite: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input text file not found: {input_path}")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        if overwrite:
            tmp_path.unlink()
        else:
            raise FileExistsError(f"Temporary output already exists: {tmp_path}")

    written = 0
    docs = 0
    batch: list[str] = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as src, tmp_path.open("wb") as dst:
        progress = tqdm(
            total=max_tokens,
            desc=f"Encoding {input_path.name}",
            unit="tok",
            disable=not sys.stderr.isatty(),
        )
        try:
            for line in src:
                if written >= max_tokens:
                    break
                batch.append(line.rstrip("\n"))
                if len(batch) >= batch_size:
                    written, docs = _flush_batch(tokenizer, batch, dst, max_tokens, written, docs, progress)
                    batch.clear()
            if batch and written < max_tokens:
                written, docs = _flush_batch(tokenizer, batch, dst, max_tokens, written, docs, progress)
        finally:
            progress.close()
    tmp_path.replace(output_path)
    metadata = {"input": str(input_path), "output": str(output_path), "tokens": written, "documents": docs, "dtype": "int32"}
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {written} tokens from {docs} docs to {output_path}")


def _flush_batch(tokenizer, batch: list[str], dst, max_tokens: int, written: int, docs: int, progress) -> tuple[int, int]:
    """Encode and write one batch of documents as a single contiguous token array.

    Args:
        tokenizer: Tokenizer used to encode text documents.
        batch: Text documents to encode.
        dst: Binary output file handle.
        max_tokens: Maximum number of tokens to write across the whole output.
        written: Number of tokens already written.
        docs: Number of documents already consumed.
        progress: Progress bar updated once per flushed batch.

    Returns:
        Updated ``(written, docs)`` counters.
    """

    encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
    eos = getattr(tokenizer, "eos_token_id", None)
    remaining = max_tokens - written
    if remaining <= 0:
        return written, docs

    output_ids: list[int] = []
    batch_docs = 0
    for ids in encoded:
        if remaining <= 0:
            break
        ids = ids if eos is None else [*ids, int(eos)]
        if len(ids) > remaining:
            ids = ids[:remaining]
        output_ids.extend(ids)
        remaining -= len(ids)
        batch_docs += 1

    if not output_ids:
        return written, docs

    array = np.asarray(output_ids, dtype=np.int32)
    array.tofile(dst)
    written += int(array.size)
    docs += batch_docs
    progress.update(int(array.size))
    return written, docs


if __name__ == "__main__":
    main()
