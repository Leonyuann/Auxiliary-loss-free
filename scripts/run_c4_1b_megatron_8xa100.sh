#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if command -v uv >/dev/null 2>&1; then
  train_cmd=(uv run torchrun)
elif [[ -x ".venv/bin/torchrun" ]]; then
  train_cmd=(.venv/bin/torchrun)
else
  echo "Neither uv nor .venv torchrun commands are available." >&2
  exit 1
fi

train_token_file="${C4_TRAIN_TOKEN_FILE:-/vepfs-mlp2/ylq/data/c4/c4_train_owt_bpe32k_tokens.i32}"
validation_token_file="${C4_VALIDATION_TOKEN_FILE:-/vepfs-mlp2/ylq/data/c4/c4_validation_owt_bpe32k_tokens.i32}"

nproc_per_node="${NPROC_PER_NODE:-8}"
max_steps="${MAX_STEPS:-100000}"
block_size="${BLOCK_SIZE:-512}"
micro_batch_size="${MICRO_BATCH_SIZE:-2}"
global_batch_size="${GLOBAL_BATCH_SIZE:-16}"
grad_accum="${GRADIENT_ACCUMULATION_STEPS:-4}"
learning_rate="${LEARNING_RATE:-${LR:-3e-4}}"
weight_decay="${WEIGHT_DECAY:-0.1}"
warmup_steps="${WARMUP_STEPS:-1000}"
scheduler_type="${SCHEDULER_TYPE:-cosine}"
max_grad_norm="${MAX_GRAD_NORM:-1.0}"

wandb_enabled="${WANDB_ENABLED:-true}"
wandb_entity="${WANDB_ENTITY:-liangqingyuann-huazhong-university-of-science-and-technology}"
wandb_project="${WANDB_PROJECT:-Load-balance}"
wandb_group="${WANDB_GROUP:-c4-1b-megatron}"

common_overrides=(
  --training.max_steps "$max_steps"
  --training.batch_size "$micro_batch_size"
  --training.gradient_accumulation_steps "$grad_accum"
  --training.learning_rate "$learning_rate"
  --training.weight_decay "$weight_decay"
  --training.warmup_steps "$warmup_steps"
  --training.scheduler_type "$scheduler_type"
  --training.max_grad_norm "$max_grad_norm"
  --data.block_size "$block_size"
  --data.train_files "$train_token_file"
  --data.validation_files "$validation_token_file"
  --megatron.micro_batch_size "$micro_batch_size"
  --megatron.global_batch_size "$global_batch_size"
  --wandb.enabled "$wandb_enabled"
  --wandb.entity "$wandb_entity"
  --wandb.project "$wandb_project"
  --wandb.group "$wandb_group"
)

torchrun_args=(--standalone --nproc_per_node="$nproc_per_node" -m alf.megatron_train)

if [[ "${RUN_ALF:-1}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_1b_megatron_alf.py "${common_overrides[@]}"
fi

if [[ "${RUN_EMA:-1}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_1b_megatron_alf_ema.py \
    "${common_overrides[@]}" \
    --alf.bias_ema_beta "${ALF_EMA_BETA:-0.5}" \
    --alf.bias_update_rate "${ALF_EMA_RATE:-1e-1}"
fi

if [[ "${RUN_AUX:-0}" == "1" ]]; then
  "${train_cmd[@]}" "${torchrun_args[@]}" experiments/qwen3_moe_c4_1b_megatron_aux_loss.py "${common_overrides[@]}"
fi
