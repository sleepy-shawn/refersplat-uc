# Fisher UC Classifier Reading Guide

This directory collects the uncertainty-specific code for the Fisher UC +
present/absent classifier branch. The training and evaluation entry points are
still the original ReferSplat scripts, but the core UC logic is here.

## 1. Fisher uncertainty

`fisher.py` contains the shared math:

```text
F_i = per-Gaussian Fisher energy
u_i = -log(F_i + eps)
u_rank_i = rank01(u_i)
```

The main experiment uses `color_uncertainty_rank01`, where larger values mean
the Gaussian is less sensitive under the RGB/color Fisher probe and therefore
more uncertain.

Producing the tensor is still done by:

```bash
python scripts/compute_rgb_param_fisher_uncertainty.py ...
```

That script saves `rgb_param_fisher_uncertainty.pt`.

## 2. Top-k Gaussian + UC tokens

`gaussian_tokens.py` contains the exact concat logic used by the classifier:

```text
attr_i = [
  g_i, normalized_xyz_i, scale_i, rotation_i, opacity_i, SH_i, uc_i
]
```

Shape:

```text
attr_topk: [K, 128 + 3 + 3 + 4 + 1 + color_dim + 1]
```

The final `+1` is the per-Gaussian uncertainty scalar.

Important: this branch appends UC as one extra token dimension. It does not add
UC to every dimension of the 128D ReferSplat feature.

## 3. Present/absent classifier

`present_classifier.py` contains:

- `GaussianAttrConvPoolFormerHead`: maps top-k Gaussian+UC tokens to one 128D
  feature with MLP, Conv1d pooling, and a small CLS transformer.
- `PresentHead`: maps that 128D feature to one binary logit.

Default decision at evaluation:

```text
present_logit > 0  -> keep the ReferSplat mask
present_logit <= 0 -> output an empty mask
```

## 4. Integration points in the original code

- `scene/gaussian_model.py`
  - creates `GaussianAttrConvPoolFormerHead` and `PresentHead`
  - loads external Fisher UC from `--external_gaussian_uncertainty_path`

- `gaussian_renderer/__init__.py`
  - selects the same score-ranked top-k Gaussians as ReferSplat
  - calls `build_gaussian_attr_tokens(...)`
  - sends tokens into the classifier

- `train.py`
  - with `--present_head_only`, freezes base ReferSplat and trains only the
    classifier modules

- `test_metrics.py`
  - applies the present/absent gate before reporting mask metrics

## 5. One-command pipeline

The runnable experiment wrapper is:

```bash
bash scripts/run_fisher_uc_attrconv_pipeline.sh ...
```

It computes Fisher UC if missing, trains the classifier-only stage, and evaluates
the resulting present/absent gate.
