"""Tests for packed local text datasets."""

from alf.data import build_packed_text_dataset, causal_lm_collate


class DummyTokenizer:
    """Whitespace tokenizer for tests.

    Attributes:
        eos_token_id: End-of-sequence token id.
    """

    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        """Encode whitespace tokens into stable integer ids.

        Args:
            text: Input text.
            add_special_tokens: Ignored test option.

        Returns:
            Tokenizer-style dictionary.
        """

        del add_special_tokens
        return {"input_ids": [index + 1 for index, _ in enumerate(text.split())]}


def test_build_packed_text_dataset() -> None:
    """Build fixed-length token blocks from a local text fixture."""

    dataset = build_packed_text_dataset(
        DummyTokenizer(),
        ["tests/fixtures/tiny_corpus.txt"],
        block_size=4,
        max_train_samples=2,
    )

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].shape[0] == 4


def test_causal_lm_collate() -> None:
    """Collate fixed-length causal LM examples."""

    dataset = build_packed_text_dataset(DummyTokenizer(), ["tests/fixtures/tiny_corpus.txt"], block_size=4)
    batch = causal_lm_collate([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
