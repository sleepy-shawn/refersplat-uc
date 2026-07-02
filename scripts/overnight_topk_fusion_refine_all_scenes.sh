#!/usr/bin/env bash
# Overnight queue: finish current ramen refine run, then train/eval/diagnose
# the same four refined top-k fusion ablations on the remaining scenes.
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

ROOT=/data1/shuting/audioRef/output
MASTER_LOG="$ROOT/overnight_topk_fusion_refine_all_scenes.log"
CKPT="${CKPT:-chkpnt_cbasetea2519.pth}"
TEST_SEED="${TEST_SEED:-42}"
MIN_FREE_MIB="${MIN_FREE_MIB:-7000}"

# GPU0 currently often has little free memory on this machine. Try it first,
# then fall back to GPU1 so the queue does not stall all night.
GPU_Q="${GPU_Q:-0,1}"
GPU_STOPG="${GPU_STOPG:-2}"
GPU_LAM05="${GPU_LAM05:-3}"
GPU_LAM025="${GPU_LAM025:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
RUN_RAMEN_EVAL="${RUN_RAMEN_EVAL:-1}"
SCENE_LIST="${SCENE_LIST:-figurines teatime waldo_kitchen}"
RESET_LOG="${RESET_LOG:-1}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

wait_for_pids() {
  local pid
  for pid in "$@"; do
    [[ -n "$pid" ]] || continue
    while kill -0 "$pid" 2>/dev/null; do
      log "WAIT pid=$pid before overnight queue"
      sleep 300
    done
  done
}

wait_for_gpu_memory() {
  local gpu_spec="$1"
  local gpu
  local free_mib
  IFS=',' read -r -a gpu_candidates <<< "$gpu_spec"
  while true; do
    for gpu in "${gpu_candidates[@]}"; do
      free_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
        SELECTED_GPU="$gpu"
        return
      fi
    done
    log "WAIT gpu_candidates=$gpu_spec need=${MIN_FREE_MIB}MiB"
    sleep 300
  done
}

run_train() {
  local scene="$1"
  local src="$2"
  local start_ckpt="$3"
  local gpu_spec="$4"
  local suffix="$5"
  local lambda_classifier="$6"
  shift 6

  local name="${scene}_nt15_qnt_topk_fusion_${suffix}_nofp"
  local out="$ROOT/$name"
  mkdir -p "$out"
  if [[ -f "$out/$CKPT" ]]; then
    log "SKIP train $name checkpoint exists"
    return
  fi
  wait_for_gpu_memory "$gpu_spec"
  local gpu="$SELECTED_GPU"
  log "TRAIN $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    -s "$src" -m "$out" \
    --start_checkpoint "$start_ckpt" \
    --total_iters 45000 \
    --training_neg_variants attribute,category,spatial,borrow \
    --training_neg_target_ratio 0.15 \
    --lambda_com 0.1 \
    --lambda_classifier "$lambda_classifier" \
    --use_present_head \
    --use_q_nt \
    --q_nt_no_fp \
    --q_nt_num_queries 1 \
    --q_nt_pool first \
    --use_topk_evidence_fusion \
    "$@" \
    > "$out/train.log" 2>&1
  log "DONE train $name"
}

wait_all() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

eval_one() {
  local scene="$1"
  local src="$2"
  local suffix="$3"
  shift 3

  local name="${scene}_nt15_qnt_topk_fusion_${suffix}_nofp"
  local out="$ROOT/$name"
  local variant
  for variant in borrow spatial attribute category; do
    wait_for_gpu_memory "$EVAL_GPU"
    local gpu="$SELECTED_GPU"
    log "EVAL $name $variant on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python test_metrics.py \
      -s "$src" -m "$out" \
      --checkpoint_name "$CKPT" \
      --perturb_variant "$variant" \
      --test_neg_target_ratio 0.15 \
      --test_seed "$TEST_SEED" \
      --use_present_head \
      --use_q_nt \
      --q_nt_no_fp \
      --q_nt_num_queries 1 \
      --q_nt_pool first \
      --use_topk_evidence_fusion \
      "$@" \
      > "$out/eval_${variant}.log" 2>&1
  done
}

diag_one() {
  local scene="$1"
  local src="$2"
  local suffix="$3"
  shift 3

  local name="${scene}_nt15_qnt_topk_fusion_${suffix}_nofp"
  local out="$ROOT/$name"
  local variant
  for variant in borrow spatial attribute category; do
    wait_for_gpu_memory "$EVAL_GPU"
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
      --use_q_nt \
      --q_nt_no_fp \
      --q_nt_num_queries 1 \
      --q_nt_pool first \
      --use_topk_evidence_fusion \
      "$@" \
      > "$out/diag_${variant}.log" 2>&1
  done
}

run_scene() {
  local scene="$1"
  local src="$2"
  local start_ckpt="$3"

  log "BEGIN scene=$scene"
  local pids=()
  run_train "$scene" "$src" "$start_ckpt" "$GPU_Q" qln 1.0 --fusion_query_layer_norm &
  pids+=("$!")
  run_train "$scene" "$src" "$start_ckpt" "$GPU_STOPG" stopg 1.0 --fusion_detach_pooled_g &
  pids+=("$!")
  run_train "$scene" "$src" "$start_ckpt" "$GPU_LAM05" lam05 0.5 &
  pids+=("$!")
  wait_all "${pids[@]}"
  run_train "$scene" "$src" "$start_ckpt" "$GPU_LAM025" lam025 0.25
  log "DONE training scene=$scene"

  eval_one "$scene" "$src" qln --fusion_query_layer_norm
  eval_one "$scene" "$src" stopg --fusion_detach_pooled_g
  eval_one "$scene" "$src" lam05
  eval_one "$scene" "$src" lam025
  log "DONE eval scene=$scene"

  diag_one "$scene" "$src" qln --fusion_query_layer_norm
  diag_one "$scene" "$src" stopg --fusion_detach_pooled_g
  diag_one "$scene" "$src" lam05
  diag_one "$scene" "$src" lam025
  log "DONE diag scene=$scene"
}

if [[ "$RESET_LOG" == "1" ]]; then
  : > "$MASTER_LOG"
fi
log "START overnight queue"
log "GPUS qln=$GPU_Q stopg=$GPU_STOPG lam05=$GPU_LAM05 lam025=$GPU_LAM025 eval=$EVAL_GPU"
wait_for_pids ${WAIT_PIDS:-}

if [[ "$RUN_RAMEN_EVAL" == "1" ]]; then
  log "BEGIN ramen eval/diag"
  GPU="$EVAL_GPU" bash scripts/ramen_nt15_topk_fusion_refine_eval.sh
  GPU="$EVAL_GPU" bash scripts/ramen_nt15_topk_fusion_refine_diag.sh
  log "DONE ramen eval/diag"
fi

for scene in $SCENE_LIST; do
  case "$scene" in
    figurines)
      run_scene figurines /data1/shuting/audioRef/figurines /data1/shuting/audioRef/figurines/figurineschkpnt30000.pth
      ;;
    teatime)
      run_scene teatime /data1/shuting/audioRef/teatime /data1/shuting/audioRef/teatime/teatimechkpnt30000.pth
      ;;
    waldo_kitchen)
      run_scene waldo_kitchen /data1/shuting/audioRef/waldo_kitchen /data1/shuting/audioRef/waldo_kitchen/kitchenchkpnt30000.pth
      ;;
    *)
      log "Unknown scene=$scene"
      exit 1
      ;;
  esac
done

log "DONE overnight queue"
