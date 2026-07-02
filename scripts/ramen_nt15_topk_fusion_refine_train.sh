#!/usr/bin/env bash
# Train four refined top-k fusion ramen ablations at 15% negative ratio.
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
START_CKPT=/data1/shuting/audioRef/ramen/ramenchkpnt30000.pth
MASTER_LOG="$ROOT/ramen_nt15_topk_fusion_refine_train.log"

run_train() {
  local gpu="$1"
  local name="$2"
  local lambda_classifier="$3"
  shift 3
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
    --lambda_classifier "$lambda_classifier" \
    --use_present_head \
    --use_q_nt \
    --q_nt_no_fp \
    --q_nt_num_queries 1 \
    --q_nt_pool first \
    --use_topk_evidence_fusion \
    "$@" \
    > "$out/train.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $name" | tee -a "$MASTER_LOG"
}

: > "$MASTER_LOG"
run_train "${GPU0:-0}" ramen_nt15_qnt_topk_fusion_qln_nofp 1.0 --fusion_query_layer_norm &
run_train "${GPU1:-1}" ramen_nt15_qnt_topk_fusion_stopg_nofp 1.0 --fusion_detach_pooled_g &
run_train "${GPU2:-2}" ramen_nt15_qnt_topk_fusion_lam05_nofp 0.5 &
run_train "${GPU3:-3}" ramen_nt15_qnt_topk_fusion_lam025_nofp 0.25 &
wait
echo "[$(date +%H:%M:%S)] DONE all ramen top-k fusion refine training" | tee -a "$MASTER_LOG"
