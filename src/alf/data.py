"""Local text data loading and token block packing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length token blocks for causal language model training.

    Attributes:
        blocks: Packed token blocks. Each item is used as both input and label.
    """

    def __init__(self, blocks: Sequence[Sequence[int]]) -> None:
        """Create a packed text dataset.

        Args:
            blocks: Token id blocks with equal sequence length.
        """

        self.blocks = [torch.tensor(block, dtype=torch.long) for block in blocks]

    def __len__(self) -> int:
        """Return the number of packed examples.

        Returns:
            Number of packed token blocks.
        """

        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one causal language modeling example.

        Args:
            index: Example index.

        Returns:
            Dictionary containing ``input_ids`` and ``labels``.
        """

        input_ids = self.blocks[index]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def read_text_files(paths: Sequence[str | Path]) -> str:
    """Read and concatenate local UTF-8 text files.

    Args:
        paths: File paths to read.

    Returns:
        Concatenated text with newline separators.

    Raises:
        FileNotFoundError: If a path does not exist.
    """

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
) -> PackedTextDataset:
    """Build a fixed-length packed dataset from local text files.

    Args:
        tokenizer: Tokenizer object with a callable encoding interface.
        paths: Local text file paths.
        block_size: Number of tokens per training example.
        max_train_samples: Optional maximum number of examples.

    Returns:
        Packed text dataset.

    Raises:
        ValueError: If no full token block can be produced.
    """

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
    """Collate fixed-length causal language model examples.

    Args:
        batch: Dataset examples.

    Returns:
        Batched tensors.
    """

    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
