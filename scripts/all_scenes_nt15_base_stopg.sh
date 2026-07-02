#!/usr/bin/env bash
# Finish nt15 PresentHead baseline and top-k fusion stopG for non-ramen scenes.
# Produces train/eval/diagnostic logs for:
#   {figurines,teatime,waldo_kitchen}_nt15_base
#   {figurines,teatime,waldo_kitchen}_nt15_qnt_topk_fusion_stopg_nofp
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

ROOT=/data1/shuting/audioRef/output
MASTER_LOG="$ROOT/all_scenes_nt15_base_stopg.log"
CKPT="${CKPT:-chkpnt_cbasetea2519.pth}"
TEST_SEED="${TEST_SEED:-42}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
MAX_UTIL="${MAX_UTIL:-20}"

BASE_GPU_POOL="${BASE_GPU_POOL:-1,0,2,3}"
STOPG_GPU_POOL="${STOPG_GPU_POOL:-3,2,0,1}"
EVAL_GPU_POOL="${EVAL_GPU_POOL:-1,0,2,3}"
RESET_LOG="${RESET_LOG:-1}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

wait_for_gpu_available() {
  local gpu_spec="$1"
  local gpu free_mib util
  IFS=',' read -r -a gpu_candidates <<< "$gpu_spec"
  while true; do
    for gpu in "${gpu_candidates[@]}"; do
      free_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if [[ "$free_mib" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
          && (( free_mib >= MIN_FREE_MIB )) \
          && (( util <= MAX_UTIL )); then
        SELECTED_GPU="$gpu"
        return
      fi
    done
    log "WAIT gpu_candidates=$gpu_spec need_free=${MIN_FREE_MIB}MiB max_util=${MAX_UTIL}%"
    sleep 300
  done
}

scene_src() {
  case "$1" in
    figurines) echo /data1/shuting/audioRef/figurines ;;
    teatime) echo /data1/shuting/audioRef/teatime ;;
    waldo_kitchen) echo /data1/shuting/audioRef/waldo_kitchen ;;
    *) log "Unknown scene=$1"; exit 1 ;;
  esac
}

scene_ckpt() {
  case "$1" in
    figurines) echo /data1/shuting/audioRef/figurines/figurineschkpnt30000.pth ;;
    teatime) echo /data1/shuting/audioRef/teatime/teatimechkpnt30000.pth ;;
    waldo_kitchen) echo /data1/shuting/audioRef/waldo_kitchen/kitchenchkpnt30000.pth ;;
    *) log "Unknown scene=$1"; exit 1 ;;
  esac
}

have_four_logs() {
  local out="$1"
  local prefix="$2"
  local count
  count=$(ls "$out"/"${prefix}"_*.log 2>/dev/null | wc -l)
  [[ "$count" -ge 4 ]]
}

train_one() {
  local scene="$1"
  local kind="$2"
  local gpu_pool="$3"
  shift 3

  local src start_ckpt name out
  src=$(scene_src "$scene")
  start_ckpt=$(scene_ckpt "$scene")
  if [[ "$kind" == "base" ]]; then
    name="${scene}_nt15_base"
  else
    name="${scene}_nt15_qnt_topk_fusion_stopg_nofp"
  fi
  out="$ROOT/$name"
  mkdir -p "$out"

  if [[ -f "$out/$CKPT" ]]; then
    log "SKIP train $name checkpoint exists"
    return
  fi

  wait_for_gpu_available "$gpu_pool"
  local gpu="$SELECTED_GPU"
  log "TRAIN $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    -s "$src" -m "$out" \
    --start_checkpoint "$start_ckpt" \
    --total_iters 45000 \
    --training_neg_variants attribute,category,spatial,borrow \
    --training_neg_target_ratio 0.15 \
    --lambda_com 0.1 \
    --lambda_classifier 1.0 \
    --use_present_head \
    "$@" \
    > "$out/train.log" 2>&1
  log "DONE train $name"
}

eval_one() {
  local scene="$1"
  local kind="$2"
  shift 2

  local src name out variant
  src=$(scene_src "$scene")
  if [[ "$kind" == "base" ]]; then
    name="${scene}_nt15_base"
  else
    name="${scene}_nt15_qnt_topk_fusion_stopg_nofp"
  fi
  out="$ROOT/$name"

  if have_four_logs "$out" eval; then
    log "SKIP eval $name eval logs exist"
    return
  fi

  for variant in borrow spatial attribute category; do
    wait_for_gpu_available "$EVAL_GPU_POOL"
    local gpu="$SELECTED_GPU"
    log "EVAL $name $variant on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python test_metrics.py \
      -s "$src" -m "$out" \
      --checkpoint_name "$CKPT" \
      --perturb_variant "$variant" \
      --test_neg_target_ratio 0.15 \
      --test_seed "$TEST_SEED" \
      --use_present_head \
      "$@" \
      > "$out/eval_${variant}.log" 2>&1
  done
}

diag_one() {
  local scene="$1"
  local kind="$2"
  shift 2

  local src name out variant
  src=$(scene_src "$scene")
  if [[ "$kind" == "base" ]]; then
    name="${scene}_nt15_base"
  else
    name="${scene}_nt15_qnt_topk_fusion_stopg_nofp"
  fi
  out="$ROOT/$name"

  if have_four_logs "$out" diag; then
    log "SKIP diag $name diag logs exist"
    return
  fi

  for variant in borrow spatial attribute category; do
    wait_for_gpu_available "$EVAL_GPU_POOL"
    local gpu="$SELECTED_GPU"
    log "DIAG $name $variant on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python test_metrics.py \
      -s "$src" -m "$out" \
      --checkpoint_name "$CKPT" \
      --perturb_variant "$variant" \
      --test_neg_target_ratio 0.15 \
      --test_seed "$TEST_SEED" \
      --use_present_head \
      --diagnostic \
      --diag_tag "diag_${variant}" \
      "$@" \
      > "$out/diag_${variant}.log" 2>&1
  done
}

run_base_lane() {
  local scene
  for scene in figurines teatime waldo_kitchen; do
    train_one "$scene" base "$BASE_GPU_POOL"
    eval_one "$scene" base
    diag_one "$scene" base
  done
  log "DONE base lane"
}

run_stopg_lane() {
  local scene
  local args=(--use_q_nt --q_nt_no_fp --q_nt_num_queries 1 --q_nt_pool first --use_topk_evidence_fusion --fusion_detach_pooled_g)
  for scene in figurines teatime waldo_kitchen; do
    train_one "$scene" stopg "$STOPG_GPU_POOL" "${args[@]}"
    eval_one "$scene" stopg "${args[@]}"
    diag_one "$scene" stopg "${args[@]}"
  done
  log "DONE stopG lane"
}

if [[ "$RESET_LOG" == "1" ]]; then
  : > "$MASTER_LOG"
fi
log "START all-scenes nt15 base+stopG"
log "GPU pools base=$BASE_GPU_POOL stopg=$STOPG_GPU_POOL eval=$EVAL_GPU_POOL"

run_base_lane &
base_pid="$!"
run_stopg_lane &
stopg_pid="$!"

status=0
wait "$base_pid" || status=1
wait "$stopg_pid" || status=1
log "DONE all-scenes nt15 base+stopG status=$status"
exit "$status"
