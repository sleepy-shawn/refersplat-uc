# Fisher UC + Gaussian Attribute Classifier

This branch contains the Fisher uncertainty plus present/absent classifier code used in the Ref-LERF no-target experiments.

## Idea

The pipeline has two stages.

1. Estimate one scalar uncertainty value for each 3D Gaussian from a trained ReferSplat checkpoint.
2. Feed that uncertainty together with the top-k Gaussian evidence tokens into a classifier head that predicts whether the text query has a target in the scene.

The best historical variant used RGB/color Fisher uncertainty:

```text
F_i ~= E_v[(d RGB / d theta_i)^2]
u_i = rank01(-log(F_i + eps))
```

where `theta_i` is the selected Gaussian parameter group. The commonly used key is `color_uncertainty_rank01`: high value means lower Fisher sensitivity and therefore higher estimated uncertainty.

## Core Files

- `scripts/compute_rgb_param_fisher_uncertainty.py`
  Computes per-Gaussian RGB parameter Fisher energy with Hutchinson probes. It saves `rgb_param_fisher_uncertainty.pt`, including `color_uncertainty_rank01`, `full_raw_uncertainty_rank01`, and related diagnostics.

- `scripts/compute_color_fisher_uncertainty.py`
  A lighter color-coefficient-only Fisher estimator.

- `scripts/compute_pup_gaussian_uncertainty.py`
  Geometry/PUP Fisher uncertainty estimator. This is useful for ablations with `pup_uncertainty_rank01`.

- `scene/gaussian_model.py`
  Adds `GaussianAttrConvPoolFormerHead`, loads external Gaussian uncertainty tensors, and stores the classifier head in checkpoints.

- `gaussian_renderer/__init__.py`
  Builds the top-k Gaussian attribute sequence:

  ```text
  token_i = concat(g_i, xyz_i, scale_i, rotation_i, opacity_i, SH_i, u_i)
  ```

  The sequence is sorted by the same text-response top-k evidence used by ReferSplat.

- `train.py`
  Supports `--present_head_only` so only `PresentHead` and `GaussianAttrConvPoolFormerHead` are trained during stage-2 calibration.

- `test_metrics.py`
  Evaluates present/absent classification and mask metrics with the same external uncertainty tensor.

## Classifier Head

The classifier path is:

```text
top-k Gaussians -> [g, xyz, scale, rotation, opacity, SH, UC]
                -> MLP token embedding
                -> Conv1d rank-local pooling
                -> CLS Transformer
                -> PresentHead BCE classifier
```

The final head predicts one scalar `present_logit`; `present_logit > 0` means the queried object is present, otherwise the rendered mask is suppressed as empty.

## Example

For the ramen color-Fisher run:

```bash
bash scripts/run_fisher_uc_attrconv_pipeline.sh \
  --scene ramen \
  --gpu 0 \
  --source /data1/shuting/audioRef/ramen \
  --baseline-model /data1/shuting/audioRef/output/ramen_baseline_v2 \
  --checkpoint chkpnt_cbasetea2519.pth \
  --output-root /data1/shuting/audioRef/output
```

The script first computes `rgb_param_fisher_uncertainty.pt`, then trains the Gaussian attribute classifier with:

```text
--use_present_head
--present_head_only
--use_gaussian_attr_conv_head
--external_gaussian_uncertainty_key color_uncertainty_rank01
```

## Historical Run

The strongest local historical run was:

```text
/data1/shuting/audioRef/output/ramen_attrconv_colorfisher_nt15_best2519
```

with:

```text
external_gaussian_uncertainty_key=color_uncertainty_rank01
present_head_only=True
gaussian_attr_conv_pooled_tokens=64
gaussian_attr_conv_num_layers=2
gaussian_attr_conv_num_heads=4
gaussian_attr_conv_ffn_dim=256
```
