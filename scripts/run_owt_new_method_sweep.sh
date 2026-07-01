#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if command -v uv >/dev/null 2>&1; then
  train_cmd=(uv run alf-train)
elif [[ -x ".venv/bin/alf-train" ]]; then
  train_cmd=(.venv/bin/alf-train)
else
  echo "Neither uv nor .venv/bin/alf-train is available." >&2
  exit 1
fi

EMA_BETAS="${EMA_BETAS:-0.5 0.25}"
EMA_BIAS_UPDATE_RATES="${EMA_BIAS_UPDATE_RATES:-1e-3}"
ACCUM_INTERVALS="${ACCUM_INTERVALS:-4 2}"
ACCUM_BIAS_UPDATE_RATES="${ACCUM_BIAS_UPDATE_RATES:-1e-3}"
BALANCED_TOPK_VALUES="${BALANCED_TOPK_VALUES:-1 2 3}"
BALANCED_TOPK_BIAS_UPDATE_RATES="${BALANCED_TOPK_BIAS_UPDATE_RATES:-1e-3}"
RUN_EMA_SWEEP="${RUN_EMA_SWEEP:-1}"
RUN_ACCUM_SWEEP="${RUN_ACCUM_SWEEP:-1}"
RUN_BALANCED_TOPK_SWEEP="${RUN_BALANCED_TOPK_SWEEP:-1}"
PREPARE_DATA="${PREPARE_DATA:-0}"

max_steps="${MAX_STEPS:-10000}"
batch_size="${BATCH_SIZE:-128}"
wandb_enabled="${WANDB_ENABLED:-true}"
wandb_entity="${WANDB_ENTITY:-liangqingyuann-huazhong-university-of-science-and-technology}"
wandb_project="${WANDB_PROJECT:-Load-balance}"
wandb_group="${WANDB_GROUP:-owt-104m-new-method-sweep}"

if [[ "$PREPARE_DATA" == "1" ]]; then
  RUN_ALF=0 RUN_AUX=0 RUN_EMA=0 RUN_ACCUM=0 RUN_BALANCED_TOPK=0 bash scripts/run_owt_104m_baselines.sh
fi

common_overrides=(
  --training.max_steps "$max_steps"
  --training.batch_size "$batch_size"
  --wandb.enabled "$wandb_enabled"
  --wandb.entity "$wandb_entity"
  --wandb.project "$wandb_project"
  --wandb.group "$wandb_group"
)

if [[ "$RUN_EMA_SWEEP" == "1" ]]; then
  for beta in $EMA_BETAS; do
    for rate in $EMA_BIAS_UPDATE_RATES; do
      safe_beta="${beta//./p}"
      safe_rate="${rate//./p}"
      run_name="qwen3_moe_owt_104m_alf_ema_beta${safe_beta}_rate${safe_rate}"
      echo "Running $run_name"
      "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf_ema.py "${common_overrides[@]}" \
        --name "$run_name" \
        --training.output_dir "outputs/$run_name" \
        --alf.bias_ema_beta "$beta" \
        --alf.bias_update_rate "$rate" \
        --wandb.tags "alf,alf-ema,owt,104m,bpe32k,sweep,beta-$beta,rate-$rate"
    done
  done
fi

if [[ "$RUN_ACCUM_SWEEP" == "1" ]]; then
  for interval in $ACCUM_INTERVALS; do
    for rate in $ACCUM_BIAS_UPDATE_RATES; do
      safe_rate="${rate//./p}"
      run_name="qwen3_moe_owt_104m_alf_accum_interval${interval}_rate${safe_rate}"
      echo "Running $run_name"
      "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf_accumulated_sign.py "${common_overrides[@]}" \
        --name "$run_name" \
        --training.output_dir "outputs/$run_name" \
        --alf.update_interval "$interval" \
        --alf.bias_update_rate "$rate" \
        --wandb.tags "alf,alf-accumulated-sign,owt,104m,bpe32k,sweep,interval-$interval,rate-$rate"
    done
  done
fi

if [[ "$RUN_BALANCED_TOPK_SWEEP" == "1" ]]; then
  for topk in $BALANCED_TOPK_VALUES; do
    for rate in $BALANCED_TOPK_BIAS_UPDATE_RATES; do
      safe_rate="${rate//./p}"
      run_name="qwen3_moe_owt_104m_alf_balanced_topk${topk}_rate${safe_rate}"
      echo "Running $run_name"
      "${train_cmd[@]}" experiments/qwen3_moe_owt_104m_alf_balanced_topk_sign.py "${common_overrides[@]}" \
        --name "$run_name" \
        --training.output_dir "outputs/$run_name" \
        --alf.bias_update_topk "$topk" \
        --alf.bias_update_rate "$rate" \
        --wandb.tags "alf,alf-balanced-topk-sign,owt,104m,bpe32k,sweep,topk-$topk,rate-$rate"
    done
  done
fi
