#!/usr/bin/env bash
# Train four ramen no-target classification experiments at 15% negative ratio.
#
# Experiments:
#   1. ramen_nt15_base              PresentHead baseline
#   2. ramen_nt15_qnt1              original q_nt, one no-target query
#   3. ramen_nt15_qnt4_gap          four q_nt queries, GAP before PresentHead
#   4. ramen_nt15_qnt_scene_fusion  q_nt + scene-aware fusion before PresentHead
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

GPU="${GPU:-0}"
SRC=/data1/shuting/audioRef/ramen
START_CKPT="$SRC/ramenchkpnt30000.pth"
ROOT=/data1/shuting/audioRef/output
MASTER_LOG="$ROOT/ramen_nt15_4exp_train.log"

run_train() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "[$(date)] TRAIN $name on GPU $GPU" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    -s "$SRC" -m "$out" \
    --start_checkpoint "$START_CKPT" \
    --total_iters 45000 \
    --training_neg_variants attribute,category,spatial,borrow \
    --training_neg_target_ratio 0.15 \
    --lambda_com 0.1 \
    --lambda_classifier 1.0 \
    --use_present_head \
    "$@" \
    > "$out/train.log" 2>&1
  echo "[$(date)] DONE  $name" | tee -a "$MASTER_LOG"
}

: > "$MASTER_LOG"
run_train ramen_nt15_base
run_train ramen_nt15_qnt1 --use_q_nt --q_nt_num_queries 1 --q_nt_pool first
run_train ramen_nt15_qnt4_gap --use_q_nt --q_nt_num_queries 4 --q_nt_pool gap
run_train ramen_nt15_qnt_scene_fusion --use_q_nt --q_nt_num_queries 1 --q_nt_pool first --use_scene_aware_fusion
