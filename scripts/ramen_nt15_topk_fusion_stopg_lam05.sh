#!/usr/bin/env bash
# Train/evaluate ramen top-k fusion stopG with a weaker classifier loss.
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

GPU="${GPU:-2}"
SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
START_CKPT=/data1/shuting/audioRef/ramen/ramenchkpnt30000.pth
CKPT="${CKPT:-chkpnt_cbasetea2519.pth}"
TEST_SEED="${TEST_SEED:-42}"
NAME=ramen_nt15_qnt_topk_fusion_stopg_lam05_nofp
OUT="$ROOT/$NAME"
MASTER_LOG="$ROOT/${NAME}.queue.log"

COMMON_EVAL_ARGS=(
  -s "$SRC" -m "$OUT"
  --checkpoint_name "$CKPT"
  --test_neg_target_ratio 0.15
  --test_seed "$TEST_SEED"
  --use_present_head
  --use_q_nt
  --q_nt_no_fp
  --q_nt_num_queries 1
  --q_nt_pool first
  --use_topk_evidence_fusion
  --fusion_detach_pooled_g
)

mkdir -p "$OUT"
: > "$MASTER_LOG"

echo "[$(date +%H:%M:%S)] TRAIN $NAME on GPU $GPU lambda_classifier=0.5" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  -s "$SRC" -m "$OUT" \
  --start_checkpoint "$START_CKPT" \
  --total_iters 45000 \
  --training_neg_variants attribute,category,spatial,borrow \
  --training_neg_target_ratio 0.15 \
  --lambda_com 0.1 \
  --lambda_classifier 0.5 \
  --use_present_head \
  --use_q_nt \
  --q_nt_no_fp \
  --q_nt_num_queries 1 \
  --q_nt_pool first \
  --use_topk_evidence_fusion \
  --fusion_detach_pooled_g \
  > "$OUT/train.log" 2>&1
echo "[$(date +%H:%M:%S)] DONE train $NAME" | tee -a "$MASTER_LOG"

for variant in borrow spatial attribute category; do
  echo "[$(date +%H:%M:%S)] EVAL $NAME $variant on GPU $GPU" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python test_metrics.py \
    "${COMMON_EVAL_ARGS[@]}" \
    --perturb_variant "$variant" \
    > "$OUT/eval_${variant}.log" 2>&1

  echo "[$(date +%H:%M:%S)] DIAG $NAME $variant on GPU $GPU" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python test_metrics.py \
    "${COMMON_EVAL_ARGS[@]}" \
    --perturb_variant "$variant" \
    --diagnostic \
    --diag_tag "diag_${variant}" \
    > "$OUT/diag_${variant}.log" 2>&1
done

echo "[$(date +%H:%M:%S)] DONE $NAME eval+diag" | tee -a "$MASTER_LOG"
