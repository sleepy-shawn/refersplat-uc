#!/usr/bin/env bash
# Train the next three ramen top-k fusion experiments:
# 1) query-LN + stop-gradient pooled_g
# 2) stop-gradient pooled_g with heavier no-target classifier weight
# 3) two-stage top-k fusion: reuse/train stage1, then PresentHead-only calibration
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
START_CKPT=/data1/shuting/audioRef/ramen/ramenchkpnt30000.pth
MASTER_LOG="$ROOT/ramen_nt15_topk_fusion_next_train.log"

run_train() {
  local gpu="$1"
  local name="$2"
  local start_ckpt="$3"
  local total_iters="$4"
  local neg_ratio="$5"
  shift 5

  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] TRAIN $name on GPU $gpu start=$start_ckpt total_iters=$total_iters neg_ratio=$neg_ratio" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    -s "$SRC" -m "$out" \
    --start_checkpoint "$start_ckpt" \
    --total_iters "$total_iters" \
    --training_neg_variants attribute,category,spatial,borrow \
    --training_neg_target_ratio "$neg_ratio" \
    --lambda_com 0.1 \
    --lambda_classifier 1.0 \
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

run_train "${GPU0:-1}" ramen_nt15_qnt_topk_fusion_qln_stopg_nofp "$START_CKPT" 45000 0.15 \
  --fusion_query_layer_norm --fusion_detach_pooled_g &

run_train "${GPU1:-2}" ramen_nt15_qnt_topk_fusion_stopg_negw2_nofp "$START_CKPT" 45000 0.15 \
  --fusion_detach_pooled_g --present_negative_weight 2.0 &

(
  stage1=ramen_nt15_qnt_topk_fusion_nofp
  stage1_ckpt="$ROOT/$stage1/chkpnt_cbasetea2519.pth"
  if [[ ! -f "$stage1_ckpt" ]]; then
    stage1=ramen_nt15_qnt_topk_fusion_2stage_stage1_nofp
    stage1_ckpt="$ROOT/$stage1/chkpnt_cbasetea2519.pth"
    run_train "${GPU2:-3}" "$stage1" "$START_CKPT" 45000 0.15
  else
    echo "[$(date +%H:%M:%S)] REUSE stage1 $stage1_ckpt" | tee -a "$MASTER_LOG"
  fi
  run_train "${GPU2:-3}" ramen_nt15_qnt_topk_fusion_2stage_nofp "$stage1_ckpt" 50000 0.50 \
    --present_head_only
) &

wait
echo "[$(date +%H:%M:%S)] DONE all ramen next top-k fusion training" | tee -a "$MASTER_LOG"
