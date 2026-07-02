#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Fisher uncertainty + Gaussian attribute classifier.

Required:
  --scene NAME              Scene name, e.g. ramen
  --source PATH             Ref-LERF scene path
  --baseline-model PATH     Existing ReferSplat model directory

Optional:
  --checkpoint NAME         Checkpoint filename in baseline model
  --output-root PATH        Root directory for outputs
  --gpu ID                  CUDA_VISIBLE_DEVICES id
  --iters N                 Stage-2 classifier iterations
  --uc-key KEY              External uncertainty key

Example:
  bash scripts/run_fisher_uc_attrconv_pipeline.sh \
    --scene ramen \
    --gpu 0 \
    --source /data1/shuting/audioRef/ramen \
    --baseline-model /data1/shuting/audioRef/output/ramen_baseline_v2 \
    --checkpoint chkpnt_cbasetea2519.pth \
    --output-root /data1/shuting/audioRef/output
EOF
}

SCENE=""
SRC=""
BASE_MODEL=""
CKPT="chkpnt_cbasetea2519.pth"
OUT_ROOT="/data1/shuting/audioRef/output"
GPU="0"
ITERS="45000"
UC_KEY="color_uncertainty_rank01"
TRAIN_VARIANTS="attribute,category,spatial,borrow"
EVAL_VARIANTS=(attribute category spatial borrow)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --source) SRC="$2"; shift 2 ;;
    --baseline-model) BASE_MODEL="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --output-root) OUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --uc-key) UC_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$SCENE" || -z "$SRC" || -z "$BASE_MODEL" ]]; then
  echo "--scene, --source, and --baseline-model are required." >&2
  usage
  exit 2
fi

if [[ -f /home/shuting/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/shuting/miniconda3/etc/profile.d/conda.sh
  conda activate refsplat
fi

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$OUT_ROOT"

CKPT_PATH="$BASE_MODEL/$CKPT"
FISHER_DIR="$BASE_MODEL/rgb_param_fisher_uncertainty/${CKPT%.pth}_hutch1"
FISHER_PT="$FISHER_DIR/rgb_param_fisher_uncertainty.pt"
OUT="$OUT_ROOT/${SCENE}_attrconv_colorfisher_nt15_best2519"

if [[ ! -f "$FISHER_PT" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] compute Fisher UC -> $FISHER_PT"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/compute_rgb_param_fisher_uncertainty.py \
    -s "$SRC" \
    -m "$BASE_MODEL" \
    --checkpoint_name "$CKPT" \
    --output_dir "$FISHER_DIR" \
    --hutchinson_samples 1 \
    --max_train_cameras -1 \
    --camera_stride 1
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] reuse Fisher UC: $FISHER_PT"
fi

mkdir -p "$OUT"

COMMON_ARGS=(
  --use_present_head
  --present_head_only
  --present_head_dropout 0.0
  --use_gaussian_attr_conv_head
  --external_gaussian_uncertainty_path "$FISHER_PT"
  --external_gaussian_uncertainty_key "$UC_KEY"
  --gaussian_attr_conv_pooled_tokens 64
  --gaussian_attr_conv_num_layers 2
  --gaussian_attr_conv_num_heads 4
  --gaussian_attr_conv_ffn_dim 256
  --gaussian_attr_conv_dropout 0.0
  --gaussian_attr_conv_kernel_size 5
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] train classifier -> $OUT"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  -s "$SRC" \
  -m "$OUT" \
  --start_checkpoint "$CKPT_PATH" \
  --total_iters "$ITERS" \
  --reset_iter_on_restore \
  --training_neg_variants "$TRAIN_VARIANTS" \
  --training_neg_target_ratio 0.15 \
  --lambda_classifier 1.0 \
  "${COMMON_ARGS[@]}" \
  > "$OUT/train.log" 2>&1

for variant in "${EVAL_VARIANTS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval $variant"
  CUDA_VISIBLE_DEVICES="$GPU" python test_metrics.py \
    -s "$SRC" \
    -m "$OUT" \
    --checkpoint_name "$CKPT" \
    --perturb_variant "$variant" \
    --test_neg_target_ratio 0.15 \
    --test_seed 42 \
    "${COMMON_ARGS[@]}" \
    > "$OUT/eval_${variant}.log" 2>&1
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] done: $OUT"
