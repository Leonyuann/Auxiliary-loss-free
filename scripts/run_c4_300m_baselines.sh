#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if command -v uv >/dev/null 2>&1; then
  python_cmd=(uv run python)
  train_cmd=(uv run torchrun)
elif [[ -x ".venv/bin/python" && -x ".venv/bin/torchrun" ]]; then
  python_cmd=(.venv/bin/python)
  train_cmd=(.venv/bin/torchrun)
else
  echo "Neither uv nor .venv torchrun commands are available." >&2
  exit 1
fi

c4_dir="${C4_DIR:-/vepfs-mlp2/ylq/data/c4/en}"
tokenizer_dir="${TOKENIZER_DIR:-/vepfs-mlp2/ylq/tokenizers/owt_bpe_32k}"
train_token_file="${C4_TRAIN_TOKEN_FILE:-/vepfs-mlp2/ylq/data/c4/c4_train_owt_bpe32k_tokens.i32}"
validation_token_file="${C4_VALIDATION_TOKEN_FILE:-/vepfs-mlp2/ylq/data/c4/c4_validation_owt_bpe32k_tokens.i32}"

nproc_per_node="${NPROC_PER_NODE:-2}"
block_size="${BLOCK_SIZE:-512}"
batch_size="${BATCH_SIZE:-128}"
grad_accum="${GRADIENT_ACCUMULATION_STEPS:-2}"
seed="${SEED:-42}"
max_steps="${MAX_STEPS:-20000}"
train_tokens="${C4_TRAIN_TOKENS:-6000000000}"
validation_tokens="${C4_VALIDATION_TOKENS:-16777216}"

wandb_enabled="${WANDB_ENABLED:-true}"
wandb_entity="${WANDB_ENTITY:-liangqingyuann-huazhong-university-of-science-and-technology}"
wandb_project="${WANDB_PROJECT:-Load-balance}"
wandb_group="${WANDB_GROUP:-c4-300m}"

prepare_args=()
if [[ "${C4_OVERWRITE:-0}" == "1" ]]; then
  prepare_args+=(--overwrite)
fi

if [[ "${RUN_PREPARE:-0}" == "1" ]]; then
  "${python_cmd[@]}" scripts/prepare_c4_bpe_tokens.py \
    --c4-dir "$c4_dir" \
    --tokenizer-dir "$tokenizer_dir" \
    --train-output "$train_token_file" \
    --validation-output "$validation_token_file" \
    --max-train-tokens "$train_tokens" \
    --max-validation-tokens "$validation_tokens" \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-8192}" \
    "${prepare_args[@]}"
fi

common_overrides=(
  --training.max_steps "$max_steps"
  --training.batch_size "$batch_size"
  --training.gradient_accumulation_steps "$grad_accum"
  --training.seed "$seed"
  --data.block_size "$block_size"
  --data.train_files "$train_token_file"
  --data.validation_files "$validation_token_file"
  --wandb.enabled "$wandb_enabled"
  --wandb.entity "$wandb_entity"
  --wandb.project "$wandb_project"
  --wandb.group "$wandb_group"
)

variance_ema_overrides=(
  --alf.bias_adaptive_beta_min "${ALF_ADAPTIVE_BETA_MIN:-0.1}"
  --alf.bias_adaptive_beta_max "${ALF_ADAPTIVE_BETA_MAX:-0.95}"
  --alf.bias_adaptive_variance_reference "${ALF_ADAPTIVE_VARIANCE_REFERENCE:-2.5e-3}"
  --alf.bias_adaptive_state_decay "${ALF_ADAPTIVE_STATE_DECAY:-0.9}"
)

persistent_ema_overrides=(
  --alf.bias_adaptive_beta_min "${ALF_ADAPTIVE_BETA_MIN:-0.25}"
  --alf.bias_adaptive_beta_max "${ALF_ADAPTIVE_BETA_MAX:-0.75}"
  --alf.bias_adaptive_variance_reference "${ALF_ADAPTIVE_VARIANCE_REFERENCE:-2.5e-3}"
  --alf.bias_adaptive_state_decay "${ALF_ADAPTIVE_STATE_DECAY:-0.9}"
)

torchrun_args=(--standalone --nproc_per_node="$nproc_per_node" -m alf.train)

if [[ "${RUN_ALF:-1}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_300m_alf.py "${common_overrides[@]}"
fi

if [[ "${RUN_EMA:-1}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_300m_alf_ema.py \
    "${common_overrides[@]}" \
    --alf.bias_ema_beta "${ALF_EMA_BETA:-0.5}" \
    --alf.bias_update_rate "${ALF_EMA_RATE:-1e-1}"
fi

if [[ "${RUN_ADAPTIVE_EMA_VARIANCE:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_300m_alf_adaptive_ema_variance.py \
    "${common_overrides[@]}" \
    "${variance_ema_overrides[@]}" \
    --alf.bias_update_rate "${ALF_VARIANCE_EMA_RATE:-5e-2}"
fi

if [[ "${RUN_ADAPTIVE_EMA_PERSISTENT_OSCILLATION:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_300m_alf_adaptive_ema_persistent_oscillation.py \
    "${common_overrides[@]}" \
    "${persistent_ema_overrides[@]}" \
    --alf.bias_update_rate "${ALF_ADAPTIVE_EMA_RATE:-1e-1}"
fi

if [[ "${RUN_ADAPTIVE_EMA_GAIN_COUPLED:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_300m_alf_adaptive_ema_gain_coupled.py \
    "${common_overrides[@]}" \
    "${persistent_ema_overrides[@]}" \
    --alf.bias_update_rate "${ALF_ADAPTIVE_EMA_RATE:-1e-1}" \
    --alf.bias_gain_coupled_normalized_gain "${ALF_NORMALIZED_GAIN:-0.03333333333333333}" \
    --alf.bias_gain_coupled_rate_min "${ALF_GAIN_RATE_MIN:-0.05}" \
    --alf.bias_gain_coupled_rate_max "${ALF_GAIN_RATE_MAX:-0.3}"
fi

if [[ "${RUN_AUX:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_300m_aux_loss.py "${common_overrides[@]}"
fi
