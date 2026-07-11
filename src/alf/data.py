"""Local text and pre-tokenized file datasets for causal language modeling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

TOKEN_FILE_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    ".i16": (torch.int16, 2),
    ".u16": (torch.uint16, 2),
    ".i32": (torch.int32, 4),
    ".u32": (torch.int32, 4),
    ".bin": (torch.int32, 4),
}


class PackedTextDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length token blocks for causal language model training."""

    def __init__(self, blocks: Sequence[Sequence[int]]) -> None:
        self.blocks = [torch.tensor(block, dtype=torch.long) for block in blocks]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.blocks[index]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class PackedTokenFileDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-mapped fixed-length blocks from one pre-tokenized token file."""

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        max_samples: int | None = None,
        include_labels: bool = True,
    ) -> None:
        """Initialize a memory-mapped token dataset.

        Args:
            path: Pre-tokenized file path.
            block_size: Tokens per returned sample.
            max_samples: Optional sample limit.
            include_labels: Whether to clone input ids into a labels field.

        Raises:
            ValueError: If shape, suffix, or size is invalid.
            FileNotFoundError: If the token file does not exist.
        """

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token file not found: {self.path}")
        suffix = self.path.suffix.lower()
        if suffix not in TOKEN_FILE_DTYPES:
            raise ValueError(f"Unsupported token file suffix {suffix!r}; expected one of {sorted(TOKEN_FILE_DTYPES)}.")
        dtype, item_size = TOKEN_FILE_DTYPES[suffix]
        file_size = self.path.stat().st_size
        if file_size % item_size != 0:
            raise ValueError(f"Token file size is not divisible by dtype size: {self.path}")
        self.block_size = int(block_size)
        self.include_labels = bool(include_labels)
        self.num_tokens = file_size // item_size
        self.tokens = torch.from_file(str(self.path), dtype=dtype, size=self.num_tokens)
        available_blocks = self.num_tokens // self.block_size
        self.num_blocks = available_blocks if max_samples is None else min(available_blocks, int(max_samples))
        if self.num_blocks <= 0:
            raise ValueError("No full token blocks were produced; lower data.block_size or provide more tokens.")

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)
        start = index * self.block_size
        input_ids = self.tokens[start : start + self.block_size].to(dtype=torch.long)
        item = {"input_ids": input_ids}
        if self.include_labels:
            item["labels"] = input_ids.clone()
        return item


def is_token_file(path: str | Path) -> bool:
    """Return whether a path looks like a supported pre-tokenized file."""

    return Path(path).suffix.lower() in TOKEN_FILE_DTYPES


def read_text_files(paths: Sequence[str | Path]) -> str:
    """Read and concatenate local UTF-8 text files."""

    texts: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Training text file not found: {path}")
        texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts)


def build_packed_text_dataset(
    tokenizer: object,
    paths: Sequence[str | Path],
    block_size: int,
    max_train_samples: int | None = None,
    include_labels: bool = True,
) -> Dataset[dict[str, torch.Tensor]]:
    """Build a fixed-length packed dataset from text or one token file.

    Args:
        tokenizer: Tokenizer used for raw text inputs.
        paths: Source text paths or one pre-tokenized file.
        block_size: Tokens per packed sample.
        max_train_samples: Optional sample limit.
        include_labels: Whether token-file items include cloned labels.

    Returns:
        Packed dataset suitable for the selected training backend.

    Raises:
        ValueError: If paths or packed output are invalid.
    """

    if not paths:
        raise ValueError("At least one data file is required.")
    if len(paths) == 1 and is_token_file(paths[0]):
        return PackedTokenFileDataset(
            paths[0],
            block_size=block_size,
            max_samples=max_train_samples,
            include_labels=include_labels,
        )
    if any(is_token_file(path) for path in paths):
        raise ValueError("Pre-tokenized datasets currently support exactly one token file path.")

    text = read_text_files(paths)
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = list(encoded["input_ids"])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))

    usable = len(token_ids) // block_size * block_size
    blocks = [token_ids[start : start + block_size] for start in range(0, usable, block_size)]
    if max_train_samples is not None:
        blocks = blocks[:max_train_samples]
    if not blocks:
        raise ValueError("No full token blocks were produced; lower data.block_size or add more text.")
    return PackedTextDataset(blocks)


def causal_lm_collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate fixed-length causal language model examples."""

    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
