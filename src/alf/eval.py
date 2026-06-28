"""Validation evaluation for ALF experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from alf.data import build_packed_text_dataset, causal_lm_collate
from alf.metrics import (
    activation_matrix_from_counts,
    activation_rows_from_counts,
    add_layer_counts,
    collect_expert_load_counts,
    loss_breakdown,
    mean_maxvio,
    serialize_activation_matrix,
)


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: object,
    config: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate validation loss, PPL, MaxVio, and expert activation.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer used for validation data.
        config: Experiment config.
        device: Evaluation device.

    Returns:
        Dictionary containing scalar metrics and expert activation structures.
    """

    dataset = build_packed_text_dataset(
        tokenizer=tokenizer,
        paths=config.data.validation_files,
        block_size=config.data.block_size,
        max_train_samples=config.data.max_validation_samples,
    )
    max_eval_samples = config.eval.max_eval_samples
    if max_eval_samples is not None:
        dataset = Subset(dataset, range(min(max_eval_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=config.eval.eval_batch_size,
        shuffle=False,
        collate_fn=causal_lm_collate,
    )

    was_training = model.training
    model.eval()
    total_lm_loss = 0.0
    total_total_loss = 0.0
    total_aux_loss = 0.0
    total_aux_loss_scaled = 0.0
    total_tokens = 0
    layer_counts: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            tokens = int((batch["labels"] != -100).sum().item())
            breakdown = loss_breakdown(outputs, model)
            total_total_loss += breakdown["loss"] * tokens
            total_lm_loss += breakdown["lm_loss"] * tokens
            total_aux_loss += breakdown["aux_loss"] * tokens
            total_aux_loss_scaled += breakdown["aux_loss_scaled"] * tokens
            total_tokens += tokens
            add_layer_counts(layer_counts, collect_expert_load_counts(model))

    if was_training:
        model.train()

    denominator = max(total_tokens, 1)
    eval_loss = total_lm_loss / denominator
    matrix, layer_names = activation_matrix_from_counts(layer_counts)
    return {
        "eval/loss": eval_loss,
        "eval/ppl": math.exp(min(eval_loss, 20.0)),
        "eval/total_loss": total_total_loss / denominator,
        "eval/aux_loss": total_aux_loss / denominator,
        "eval/aux_loss_scaled": total_aux_loss_scaled / denominator,
        "eval/maxvio_global": mean_maxvio(layer_counts),
        "eval/tokens": total_tokens,
        "eval/expert_activation_matrix": matrix,
        "eval/expert_activation_matrix_json": serialize_activation_matrix(matrix, layer_names),
        "eval/expert_activation_layers": layer_names,
        "eval/expert_activation_rows": activation_rows_from_counts(layer_counts, step=None, split="eval"),
    }
