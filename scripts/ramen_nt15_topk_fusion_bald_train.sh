#!/usr/bin/env bash
# Train two BALD-weighted ramen top-k fusion detach experiments:
# 1) stable evidence pooling: weight = 1 - BALD
# 2) uncertain evidence pooling: weight = BALD
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
START_CKPT=/data1/shuting/audioRef/ramen/ramenchkpnt30000.pth
MASTER_LOG="$ROOT/ramen_nt15_topk_fusion_bald_train.log"

run_train() {
  local gpu="$1"
  local name="$2"
  local mode="$3"

  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] TRAIN $name mode=$mode on GPU $gpu" | tee -a "$MASTER_LOG"
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
    --q_nt_num_queries 1 \
    --q_nt_pool first \
    --use_topk_evidence_fusion \
    --fusion_detach_pooled_g \
    --use_bald_evidence_weight \
    --bald_weight_mode "$mode" \
    --bald_probe_max_angle 60.0 \
    --bald_probe_strategy random \
    > "$out/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name" | tee -a "$MASTER_LOG"
}

: > "$MASTER_LOG"
run_train "${GPU0:-0}" ramen_nt15_qnt_topk_fusion_stopg_bald_stable_nofp stable &
run_train "${GPU1:-1}" ramen_nt15_qnt_topk_fusion_stopg_bald_uncertain_nofp uncertain &
wait
echo "[$(date +%H:%M:%S)] DONE all ramen BALD top-k fusion training" | tee -a "$MASTER_LOG"
