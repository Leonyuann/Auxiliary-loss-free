"""Tests for packed local text and token-file datasets."""

from array import array

from alf.data import build_packed_text_dataset, causal_lm_collate


class DummyTokenizer:
    """Whitespace tokenizer for tests."""

    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
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


def test_build_packed_i32_token_file_dataset(tmp_path) -> None:
    """Build fixed-length token blocks directly from an int32 token file."""

    token_file = tmp_path / "tokens.i32"
    with token_file.open("wb") as file:
        array("i", range(10)).tofile(file)

    dataset = build_packed_text_dataset(
        DummyTokenizer(),
        [token_file],
        block_size=4,
        max_train_samples=2,
    )

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert dataset[1]["labels"].tolist() == [4, 5, 6, 7]
