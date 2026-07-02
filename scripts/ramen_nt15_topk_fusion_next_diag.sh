#!/usr/bin/env bash
# Diagnostic pass for the next ramen top-k fusion experiments.
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

GPU="${GPU:-1}"
SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
CKPT="${CKPT:-chkpnt_cbasetea2519.pth}"
TEST_SEED="${TEST_SEED:-42}"
MASTER_LOG="$ROOT/ramen_nt15_topk_fusion_next_diag.log"

diag_one() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  for variant in borrow spatial attribute category; do
    echo "[$(date +%H:%M:%S)] DIAG $name $variant on GPU $GPU" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python test_metrics.py \
      -s "$SRC" -m "$out" \
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

: > "$MASTER_LOG"
diag_one ramen_nt15_qnt_topk_fusion_qln_stopg_nofp --fusion_query_layer_norm --fusion_detach_pooled_g
diag_one ramen_nt15_qnt_topk_fusion_stopg_negw2_nofp --fusion_detach_pooled_g
diag_one ramen_nt15_qnt_topk_fusion_2stage_nofp
echo "[$(date +%H:%M:%S)] DONE ramen next top-k fusion diagnostic" | tee -a "$MASTER_LOG"
