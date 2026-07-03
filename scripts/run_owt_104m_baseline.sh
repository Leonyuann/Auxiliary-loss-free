#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if command -v uv >/dev/null 2>&1; then
  python_cmd=(uv run python)
  train_cmd=(uv run alf-train)
elif [[ -x ".venv/bin/python" && -x ".venv/bin/alf-train" ]]; then
  python_cmd=(.venv/bin/python)
  train_cmd=(.venv/bin/alf-train)
else
  echo "Neither uv nor .venv training commands are available." >&2
  exit 1
fi

data_root="${DATA_ROOT:-/vepfs-mlp2/ylq}"
owt_train="${OWT_TRAIN_TEXT:-/xts001/data/owt_train.txt}"
owt_valid="${OWT_VALID_TEXT:-/xts001/data/owt_valid.txt}"
owt_dir="${OWT_TOKEN_DIR:-$data_root/data}"
train_token_file="${OWT_TRAIN_TOKEN_FILE:-$owt_dir/train_1310m_bpe32k_tokens.i32}"
validation_token_file="${OWT_VALIDATION_TOKEN_FILE:-$owt_dir/validation_16m_bpe32k_tokens.i32}"
tokenizer_dir="${TOKENIZER_DIR:-$data_root/tokenizers/owt_bpe_32k}"

block_size="${BLOCK_SIZE:-512}"
batch_size="${BATCH_SIZE:-128}"
max_steps="${MAX_STEPS:-10000}"
train_tokens="${OWT_TRAIN_TOKENS:-$((max_steps * batch_size * block_size))}"
validation_tokens="${OWT_VALIDATION_TOKENS:-16777216}"

wandb_enabled="${WANDB_ENABLED:-true}"
wandb_entity="${WANDB_ENTITY:-liangqingyuann-huazhong-university-of-science-and-technology}"
wandb_project="${WANDB_PROJECT:-Load-balance}"

prepare_args=()
if [[ "${OWT_OVERWRITE:-0}" == "1" ]]; then
  prepare_args+=(--overwrite)
fi
if [[ "${FORCE_TRAIN_TOKENIZER:-0}" == "1" ]]; then
  prepare_args+=(--force-train-tokenizer)
fi

"${python_cmd[@]}" scripts/prepare_text_bpe_tokens.py \
  --input "$owt_train" \
  --output "$train_token_file" \
  --tokenizer-dir "$tokenizer_dir" \
  --train-tokenizer-input "$owt_train" \
  --tokenizer-train-max-docs "${TOKENIZER_TRAIN_MAX_DOCS:-200000}" \
  --vocab-size 32768 \
  --max-tokens "$train_tokens" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-8192}" \
  "${prepare_args[@]}"

"${python_cmd[@]}" scripts/prepare_text_bpe_tokens.py \
  --input "$owt_valid" \
  --output "$validation_token_file" \
  --tokenizer-dir "$tokenizer_dir" \
  --vocab-size 32768 \
  --max-tokens "$validation_tokens" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-8192}" \
  "${prepare_args[@]}"

common_overrides=(
  --training.max_steps "$max_steps"
  --training.batch_size "$batch_size"
  --wandb.enabled "$wandb_enabled"
  --wandb.entity "$wandb_entity"
  --wandb.project "$wandb_project"
  --data.train_files "$train_token_file"
  --data.validation_files "$validation_token_file"
)

if [[ "${RUN_ALF:-1}" == "1" ]]; then
  "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf.py "${common_overrides[@]}"
fi

if [[ "${RUN_EMA:-0}" == "1" ]]; then
  "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf_ema.py \
    "${common_overrides[@]}" \
    --alf.bias_ema_beta "${ALF_EMA_BETA:-0.9}"
fi

if [[ "${RUN_ACCUM:-0}" == "1" ]]; then
  "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf_accumulated_sign.py \
    "${common_overrides[@]}" \
    --alf.update_interval "${ALF_ACCUM_INTERVAL:-16}"
fi

if [[ "${RUN_BALANCED_TOPK:-0}" == "1" ]]; then
  "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf_balanced_topk_sign.py \
    "${common_overrides[@]}" \
    --alf.bias_update_topk "${ALF_BALANCED_TOPK:-2}"
fi

if [[ "${RUN_AUX:-1}" == "1" ]]; then
  "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_aux_loss.py "${common_overrides[@]}"
fi
