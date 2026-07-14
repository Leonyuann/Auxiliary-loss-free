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

data_root="${DATA_ROOT:-/vepfs-mlp2/ylq}"
owt_train="${OWT_TRAIN_TEXT:-/xts001/data/owt_train.txt}"
owt_valid="${OWT_VALID_TEXT:-/xts001/data/owt_valid.txt}"
owt_dir="${OWT_TOKEN_DIR:-$data_root/data}"
train_token_file="${OWT_TRAIN_TOKEN_FILE:-$owt_dir/train_1310m_bpe32k_tokens.i32}"
validation_token_file="${OWT_VALIDATION_TOKEN_FILE:-$owt_dir/validation_16m_bpe32k_tokens.i32}"
tokenizer_dir="${TOKENIZER_DIR:-$data_root/tokenizers/owt_bpe_32k}"

nproc_per_node="${NPROC_PER_NODE:-1}"
grad_accum="${GRADIENT_ACCUMULATION_STEPS:-1}"
seed="${SEED:-42}"
ddp_backend="${DDP_BACKEND:-nccl}"
block_size="${BLOCK_SIZE:-512}"
batch_size="${BATCH_SIZE:-128}"
max_steps="${MAX_STEPS:-10000}"
train_tokens="${OWT_TRAIN_TOKENS:-$((max_steps * batch_size * block_size * grad_accum * nproc_per_node))}"
validation_tokens="${OWT_VALIDATION_TOKENS:-16777216}"

wandb_enabled="${WANDB_ENABLED:-true}"
wandb_entity="${WANDB_ENTITY:-liangqingyuann-huazhong-university-of-science-and-technology}"
wandb_project="${WANDB_PROJECT:-Load-balance}"
wandb_group="${WANDB_GROUP:-owt-104m}"

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
  --training.seed "$seed"
  --training.gradient_accumulation_steps "$grad_accum"
  --training.ddp_backend "$ddp_backend"
  --data.block_size "$block_size"
  --wandb.enabled "$wandb_enabled"
  --wandb.entity "$wandb_entity"
  --wandb.project "$wandb_project"
  --data.train_files "$train_token_file"
  --wandb.group "$wandb_group"
  --data.validation_files "$validation_token_file"
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
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf.py "${common_overrides[@]}"
fi

if [[ "${RUN_EMA:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_ema.py \
    "${common_overrides[@]}" \
    --alf.bias_ema_beta "${ALF_EMA_BETA:-0.5}" \
    --alf.bias_update_rate "${ALF_EMA_RATE:-1e-1}"
fi

if [[ "${RUN_ADAPTIVE_PER_EXPERT:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_adaptive_per_expert.py \
    "${common_overrides[@]}" \
    --alf.bias_update_rate "${ALF_ADAPTIVE_PER_EXPERT_BASE_RATE:-1e-3}" \
    --alf.bias_adaptive_per_expert_beta "${ALF_ADAPTIVE_PER_EXPERT_BETA:-0.9}" \
    --alf.bias_adaptive_per_expert_epsilon "${ALF_ADAPTIVE_PER_EXPERT_EPSILON:-1e-8}"
fi

if [[ "${RUN_ADAPTIVE_PER_EXPERT_MOMENTUM:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_adaptive_per_expert_momentum.py \
    "${common_overrides[@]}" \
    --alf.bias_update_rate "${ALF_ADAPTIVE_PER_EXPERT_BASE_RATE:-1e-3}" \
    --alf.bias_adaptive_per_expert_beta "${ALF_ADAPTIVE_PER_EXPERT_BETA:-0.9}" \
    --alf.bias_adaptive_per_expert_momentum_beta "${ALF_ADAPTIVE_PER_EXPERT_MOMENTUM_BETA:-0.9}" \
    --alf.bias_adaptive_per_expert_epsilon "${ALF_ADAPTIVE_PER_EXPERT_EPSILON:-1e-8}"
fi

if [[ "${RUN_ADAPTIVE_EMA_VARIANCE:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_adaptive_ema_variance.py \
    "${common_overrides[@]}" \
    "${variance_ema_overrides[@]}" \
    --alf.bias_update_rate "${ALF_VARIANCE_EMA_RATE:-1e-1}"
fi

if [[ "${RUN_ADAPTIVE_EMA_PERSISTENT_OSCILLATION:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_adaptive_ema_persistent_oscillation.py \
    "${common_overrides[@]}" \
    "${persistent_ema_overrides[@]}" \
    --alf.bias_update_rate "${ALF_ADAPTIVE_EMA_RATE:-1e-1}"
fi

if [[ "${RUN_ADAPTIVE_EMA_GAIN_COUPLED:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_adaptive_ema_gain_coupled.py \
    "${common_overrides[@]}" \
    "${persistent_ema_overrides[@]}" \
    --alf.bias_update_rate "${ALF_ADAPTIVE_EMA_RATE:-1e-1}" \
    --alf.bias_gain_coupled_normalized_gain "${ALF_NORMALIZED_GAIN:-0.03333333333333333}" \
    --alf.bias_gain_coupled_rate_min "${ALF_GAIN_RATE_MIN:-0.05}" \
    --alf.bias_gain_coupled_rate_max "${ALF_GAIN_RATE_MAX:-0.3}"
fi

if [[ "${RUN_ACCUM:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_accumulated_sign.py \
    "${common_overrides[@]}" \
    --alf.update_interval "${ALF_ACCUM_INTERVAL:-4}"
fi

if [[ "${RUN_BALANCED_TOPK:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_alf_balanced_topk_sign.py \
    "${common_overrides[@]}" \
    --alf.bias_update_topk "${ALF_BALANCED_TOPK:-2}"
fi

if [[ "${RUN_AUX:-1}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_owt_104m_aux_loss.py "${common_overrides[@]}"
fi
