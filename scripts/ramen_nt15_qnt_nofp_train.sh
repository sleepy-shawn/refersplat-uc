#!/usr/bin/env bash
# Train the three q_nt no-fp ramen ablations at 15% negative ratio.
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
START_CKPT=/data1/shuting/audioRef/ramen/ramenchkpnt30000.pth
MASTER_LOG="$ROOT/ramen_nt15_qnt_nofp_train.log"

run_train() {
  local gpu="$1"
  local name="$2"
  shift 2
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] TRAIN $name on GPU $gpu" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    -s "$SRC" -m "$out" \
    --start_checkpoint "$START_CKPT" \
    --total_iters 45000 \
    --training_neg_variants attribute,category,spatial,borrow \
    --training_neg_target_ratio 0.15 \
    --lambda_com 0.1 \
    --lambda_classifier 1.0 \
    --use_present_head \
    --use_q_nt \
    --q_nt_no_fp \
    "$@" \
    > "$out/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name" | tee -a "$MASTER_LOG"
}

: > "$MASTER_LOG"
run_train 1 ramen_nt15_qnt1_nofp --q_nt_num_queries 1 --q_nt_pool first &
run_train 2 ramen_nt15_qnt4_gap_nofp --q_nt_num_queries 4 --q_nt_pool gap &
run_train 3 ramen_nt15_qnt_scene_fusion_nofp --q_nt_num_queries 1 --q_nt_pool first --use_scene_aware_fusion &
wait
echo "[$(date +%H:%M:%S)] DONE all ramen q_nt no-fp training" | tee -a "$MASTER_LOG"
