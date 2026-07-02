#!/usr/bin/env bash
# Evaluate the four ramen no-target experiments at 15% negative ratio.
set -euo pipefail

source /home/shuting/miniconda3/etc/profile.d/conda.sh
conda activate refsplat
cd /home/shuting/gslab/ReferSplat
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

GPU="${GPU:-0}"
SRC=/data1/shuting/audioRef/ramen
ROOT=/data1/shuting/audioRef/output
CKPT="${CKPT:-chkpnt_cbasetea2519.pth}"
TEST_SEED="${TEST_SEED:-42}"
MASTER_LOG="$ROOT/ramen_nt15_4exp_eval.log"

eval_one() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  for variant in borrow spatial attribute category; do
    echo "[$(date +%H:%M:%S)] EVAL $name $variant on GPU $GPU" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python test_metrics.py \
      -s "$SRC" -m "$out" \
      --checkpoint_name "$CKPT" \
      --perturb_variant "$variant" \
      --test_neg_target_ratio 0.15 \
      --test_seed "$TEST_SEED" \
      --use_present_head \
      "$@" \
      > "$out/eval_${variant}.log" 2>&1
  done
}

: > "$MASTER_LOG"
eval_one ramen_nt15_base
eval_one ramen_nt15_qnt1 --use_q_nt --q_nt_num_queries 1 --q_nt_pool first
eval_one ramen_nt15_qnt4_gap --use_q_nt --q_nt_num_queries 4 --q_nt_pool gap
eval_one ramen_nt15_qnt_scene_fusion --use_q_nt --q_nt_num_queries 1 --q_nt_pool first --use_scene_aware_fusion
echo "[$(date +%H:%M:%S)] DONE ramen nt15 four-experiment eval" | tee -a "$MASTER_LOG"
