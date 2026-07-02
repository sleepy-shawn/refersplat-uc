"""
Evaluation script computing N-acc / T-acc / gIoU for ReferSplat with
optional uncertain-token rejection.

- N-acc: fraction of negative samples (cross-scene referent) where the
  predicted mask is empty.
- T-acc: fraction of positive samples where the predicted mask is non-empty.
- gIoU: per-sample score averaged. Negative + correctly-empty = 1.0,
  negative + wrongly-non-empty = 0.0, positive = standard IoU
  (calculate_iou from test_miou).

Diagnostic mode (`--diagnostic`): also records per-sample U values, raw
pixel counts before/after the hard gate, and attributes each empty
prediction to one of {hard_gate, pixel_thresh, both, none}. Writes a CSV
to <model_path>/<diag_tag>_diagnostic.csv and prints summary stats.
"""
import os
import re
import csv
import json
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from scene import Scene
from gaussian_renderer import (
    render_variational_language_mc,
    GaussianModel,
    pick_probe_view,
)
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from test_miou import calculate_iou


DIAG_FIELDNAMES = [
    "query_id",
    "view_idx",
    "sent_idx",
    "image_name",
    "sentence",
    "category",
    "is_present",
    "is_negative",
    "negative_type",
    "present_logit",
    "present_prob",
    "head_pred_present",
    "pred_present",
    "pred_empty",
    "mask_iou",
    "max_score",
    "topk_score_mean",
    "refer_uncertainty_mean",
    "refer_uncertainty_top_mean",
    "refer_uncertainty_rel_mean",
    "refer_uncertainty_rel_top_mean",
    "refer_uncertainty_top_std",
    "refer_uncertainty_mu_mean",
    "refer_uncertainty_mu_top_mean",
    "refer_uncertainty_sample_rel_top_mean",
    "score_sensitivity_top_mean",
    "score_reparam_sensitivity_top_mean",
    "score_target_u_top_mean",
    "score_u_spearman_top",
    "gaussian_attr_conv_topk_count",
    "gaussian_attr_conv_pre_adaptive_tokens",
    "gaussian_attr_conv_pooled_tokens",
    "gaussian_attr_conv_uc_topk_mean",
    "gaussian_attr_conv_uc_topk_std",
    "variational_language_kl",
    "variational_language_mu_norm_mean",
    "variational_language_base_norm_mean",
    "variational_language_posterior_mean_norm_mean",
    "variational_language_sigma_mean",
    "variational_language_log_sigma_mean",
    "variational_language_mean_shift_norm_mean",
    "variational_language_active_gate_ratio",
    "variational_language_active_mu_norm_mean",
    "variational_language_inactive_mu_norm_mean",
    "variational_language_active_sigma_mean",
    "variational_language_inactive_sigma_mean",
    "variational_language_base_topk_score_mean",
    "variational_language_sampled_topk_score_mean",
    "variational_language_eval_samples",
    "variational_language_prob_var_mean",
    "variational_language_prob_std_mean",
    "variational_language_predictive_entropy_mean",
    "variational_language_logit_var_mean",
    "pixels_before_gate",
    "pixels_after_gate",
    "hard_gate_triggered",
    "pixel_thresh_triggered",
    "U",
]


def _tensor_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _binary_auroc(labels, scores):
    """Rank-based AUROC with average ranks for ties."""
    pairs = [(float(s), int(y)) for y, s in zip(labels, scores)
             if s is not None and not math.isnan(float(s))]
    n = len(pairs)
    if n == 0:
        return float("nan")
    n_pos = sum(y == 1 for _, y in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(y == 1 for _, y in pairs[i:j])
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(dataset, pipeline, args):
    gaussians = GaussianModel(dataset.sh_degree,
                              use_uncertain_token=getattr(args, "use_uncertain_token", False),
                              unctoken_query_mode=getattr(args, "unctoken_query_mode", "fr_plus_fp"),
                              unctoken_arch=getattr(args, "unctoken_arch", "external"),
                              use_present_head=getattr(args, "use_present_head", False),
                              present_head_hidden=getattr(args, "present_head_hidden", 128),
                              present_head_dropout=getattr(args, "present_head_dropout", 0.1),
                              use_q_nt=getattr(args, "use_q_nt", False),
                              q_nt_num_queries=getattr(args, "q_nt_num_queries", 1),
                              q_nt_pool=getattr(args, "q_nt_pool", "first"),
                              q_nt_no_fp=getattr(args, "q_nt_no_fp", False),
                              use_scene_aware_fusion=getattr(args, "use_scene_aware_fusion", False),
                              use_topk_evidence_gap=getattr(args, "use_topk_evidence_gap", False),
                              use_topk_evidence_fusion=getattr(args, "use_topk_evidence_fusion", False),
                              fusion_layer_norm=getattr(args, "fusion_layer_norm", False),
                              fusion_query_layer_norm=getattr(args, "fusion_query_layer_norm", False),
                              fusion_detach_pooled_g=getattr(args, "fusion_detach_pooled_g", False),
                              use_bald_evidence_weight=getattr(args, "use_bald_evidence_weight", False),
                              use_refer_uncertainty=getattr(args, "use_refer_uncertainty", False),
                              use_variational_language=getattr(args, "use_variational_language", False),
                              variational_language_prior_std=getattr(args, "variational_language_prior_std", 0.0025),
                              variational_language_log_sigma_min=getattr(args, "variational_language_log_sigma_min", -5.0),
                              variational_language_log_sigma_max=getattr(args, "variational_language_log_sigma_max", 2.0),
                              use_gaussian_attr_conv_head=getattr(args, "use_gaussian_attr_conv_head", False),
                              gaussian_attr_conv_pooled_tokens=getattr(args, "gaussian_attr_conv_pooled_tokens", 64),
                              gaussian_attr_conv_num_layers=getattr(args, "gaussian_attr_conv_num_layers", 2),
                              gaussian_attr_conv_num_heads=getattr(args, "gaussian_attr_conv_num_heads", 4),
                              gaussian_attr_conv_ffn_dim=getattr(args, "gaussian_attr_conv_ffn_dim", 256),
                              gaussian_attr_conv_dropout=getattr(args, "gaussian_attr_conv_dropout", 0.0),
                              gaussian_attr_conv_kernel_size=getattr(args, "gaussian_attr_conv_kernel_size", 5))
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint = os.path.join(args.model_path, args.checkpoint_name)
    (model_params, _) = torch.load(checkpoint, map_location=f"cuda:{torch.cuda.current_device()}")
    gaussians.restore(model_params, args, mode="test")
    gaussians.maybe_load_external_gaussian_uncertainty(args)

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    views = scene.getTestCameras()

    tau = float(getattr(args, "uncertain_tau", 0.5))
    pixel_thresh = int(getattr(args, "pixel_thresh", 50))
    use_unc = bool(getattr(args, "use_uncertain_token", False))
    diagnostic = bool(getattr(args, "diagnostic", False))
    present_head_threshold = float(getattr(args, "present_head_threshold", 0.0))

    n_pos = n_neg = 0
    n_pos_correct = n_neg_correct = 0
    giou_sum = 0.0
    total = 0
    # Per-positive IoUs for the two GRES segmentation-quality metrics
    # that aren't already exposed: mIoU+ (mean IoU on positives only) and
    # P@thr (fraction of positives with IoU >= thr).
    pos_ious = []

    # Diagnostic accumulators (only used when diagnostic=True)
    diag_rows = []           # one row per sample
    diag_labels = []
    diag_present_scores = []
    # Path attribution counters (negative samples → which path made it empty?)
    neg_path_counts = {"hard_gate_only": 0, "pixel_thresh_only": 0,
                       "both": 0, "neither_wrong_nonempty": 0}
    pos_killed = {"hard_gate": 0, "pixel_thresh": 0, "both": 0,
                  "survived_correct": 0, "survived_wrong_iou0": 0}

    # ----- Optional: align test-set neg ratio with training -----
    # When --test_neg_target_ratio is set in [0,1], we subsample (view, sentence)
    # pairs so the eval set's neg/(pos+neg) ≈ target. This matches the train-set
    # subsampling done via --training_neg_target_ratio so train and eval priors
    # are aligned. The default current experiment prior is 15%; pass -1 to use
    # the native test-set ratio instead.
    test_target_ratio = float(getattr(args, "test_neg_target_ratio", 0.15))
    # --test_seed controls the randomness of pos/neg subsampling. -1 = truly
    # random (os.urandom-seeded), any non-negative int = reproducible.
    # Previous behavior was a hard-coded md5 hash seed (deterministic and
    # therefore biased to the SAME subset on every run). With the expanded
    # neg-heavy test set (~80% native neg), subsampling a different random
    # subset each run gives a more honest picture of model performance.
    test_seed = int(getattr(args, "test_seed", -1))
    if test_seed < 0:
        rng_seed = int.from_bytes(os.urandom(4), "little")
    else:
        rng_seed = test_seed
    print(f"[test_subsample_seed] {rng_seed} (use --test_seed N for reproducible runs)")
    keep_set = None  # None = keep everything (default)
    if 0.0 <= test_target_ratio <= 1.0:
        all_pos = []
        all_neg = []
        for view_idx, view in enumerate(views):
            for i in range(len(view.sentence)):
                is_neg_i = bool(view.is_negative[i]) if i < len(view.is_negative) else False
                pair = (view_idx, i)
                (all_neg if is_neg_i else all_pos).append(pair)
        n_pos_avail = len(all_pos)
        n_neg_avail = len(all_neg)
        # Compute counts to achieve target. Two cases:
        if n_pos_avail > 0 and n_neg_avail > 0:
            if test_target_ratio >= 1.0:
                # All-neg degenerate
                keep_pos, keep_neg = [], all_neg
            elif test_target_ratio == 0.0:
                keep_pos, keep_neg = all_pos, []
            else:
                # Want neg/(pos+neg) = r. So pos = neg*(1-r)/r and neg = pos*r/(1-r).
                # Pick whichever side requires SHRINKING (can't expand data).
                desired_neg_for_all_pos = int(round(test_target_ratio * n_pos_avail / (1.0 - test_target_ratio)))
                desired_pos_for_all_neg = int(round((1.0 - test_target_ratio) * n_neg_avail / test_target_ratio))
                if desired_neg_for_all_pos <= n_neg_avail:
                    # Native ratio too low (typical case): keep all pos, subsample neg
                    keep_pos = all_pos
                    rng_n = random.Random(rng_seed)
                    rng_n.shuffle(all_neg)
                    keep_neg = all_neg[:desired_neg_for_all_pos]
                else:
                    # Native ratio too high relative to target: keep all neg, subsample pos
                    keep_neg = all_neg
                    rng_p = random.Random(rng_seed ^ 0xABCDEF)
                    rng_p.shuffle(all_pos)
                    keep_pos = all_pos[:desired_pos_for_all_neg]
            keep_set = set(keep_pos) | set(keep_neg)
            achieved = len(keep_neg) / max(1, len(keep_pos) + len(keep_neg))
            print(f"[test_neg_target_ratio={test_target_ratio:.3f}] "
                  f"native pos={n_pos_avail} neg={n_neg_avail} "
                  f"→ kept pos={len(keep_pos)} neg={len(keep_neg)} "
                  f"achieved={achieved*100:.2f}%")

    with torch.no_grad():
        for view_idx, view in enumerate(views):
            for i in range(len(view.sentence)):
                if keep_set is not None and (view_idx, i) not in keep_set:
                    continue
                bald_probe_view = None
                if getattr(args, "use_bald_evidence_weight", False):
                    bald_probe_view = pick_probe_view(
                        view,
                        views,
                        max_angle_deg=float(getattr(args, "bald_probe_max_angle", 60.0)),
                        rng=random.Random(view_idx * 1000003 + i),
                        strategy=getattr(args, "bald_probe_strategy", "nearest"),
                    )
                out = render_variational_language_mc(
                    view, gaussians, pipeline, bg, args,
                    sentence=view.sentence[i],
                    probe_view=bald_probe_view,
                )
                logits = out["language_feature_image"]
                prob = torch.sigmoid(logits)
                pred_raw = (prob >= 0.5).float()
                pixels_before_gate = int(pred_raw.sum().item())

                U_val = None
                present_logit = _tensor_scalar(out.get("present_logit"))
                present_prob = None
                head_pred_present = None
                max_score = _tensor_scalar(out.get("max_score"))
                topk_score_mean = _tensor_scalar(out.get("topk_score_mean"))
                refer_uncertainty_mean = _tensor_scalar(out.get("refer_uncertainty_mean"))
                refer_uncertainty_top_mean = _tensor_scalar(out.get("refer_uncertainty_top_mean"))
                refer_uncertainty_rel_mean = _tensor_scalar(out.get("refer_uncertainty_rel_mean"))
                refer_uncertainty_rel_top_mean = _tensor_scalar(out.get("refer_uncertainty_rel_top_mean"))
                refer_uncertainty_top_std = _tensor_scalar(out.get("refer_uncertainty_top_std"))
                refer_uncertainty_mu_mean = _tensor_scalar(out.get("refer_uncertainty_mu_mean"))
                refer_uncertainty_mu_top_mean = _tensor_scalar(out.get("refer_uncertainty_mu_top_mean"))
                refer_uncertainty_sample_rel_top_mean = _tensor_scalar(out.get("refer_uncertainty_sample_rel_top_mean"))
                score_sensitivity_top_mean = _tensor_scalar(out.get("score_sensitivity_top_mean"))
                score_reparam_sensitivity_top_mean = _tensor_scalar(out.get("score_reparam_sensitivity_top_mean"))
                score_target_u_top_mean = _tensor_scalar(out.get("score_target_u_top_mean"))
                score_u_spearman_top = _tensor_scalar(out.get("score_u_spearman_top"))
                gaussian_attr_conv_topk_count = _tensor_scalar(out.get("gaussian_attr_conv_topk_count"))
                gaussian_attr_conv_pre_adaptive_tokens = _tensor_scalar(out.get("gaussian_attr_conv_pre_adaptive_tokens"))
                gaussian_attr_conv_pooled_tokens = _tensor_scalar(out.get("gaussian_attr_conv_pooled_tokens"))
                gaussian_attr_conv_uc_topk_mean = _tensor_scalar(out.get("gaussian_attr_conv_uc_topk_mean"))
                gaussian_attr_conv_uc_topk_std = _tensor_scalar(out.get("gaussian_attr_conv_uc_topk_std"))
                variational_language_kl = _tensor_scalar(out.get("variational_language_kl"))
                variational_language_mu_norm_mean = _tensor_scalar(out.get("variational_language_mu_norm_mean"))
                variational_language_base_norm_mean = _tensor_scalar(out.get("variational_language_base_norm_mean"))
                variational_language_posterior_mean_norm_mean = _tensor_scalar(out.get("variational_language_posterior_mean_norm_mean"))
                variational_language_sigma_mean = _tensor_scalar(out.get("variational_language_sigma_mean"))
                variational_language_log_sigma_mean = _tensor_scalar(out.get("variational_language_log_sigma_mean"))
                variational_language_mean_shift_norm_mean = _tensor_scalar(out.get("variational_language_mean_shift_norm_mean"))
                variational_language_active_gate_ratio = _tensor_scalar(out.get("variational_language_active_gate_ratio"))
                variational_language_active_mu_norm_mean = _tensor_scalar(out.get("variational_language_active_mu_norm_mean"))
                variational_language_inactive_mu_norm_mean = _tensor_scalar(out.get("variational_language_inactive_mu_norm_mean"))
                variational_language_active_sigma_mean = _tensor_scalar(out.get("variational_language_active_sigma_mean"))
                variational_language_inactive_sigma_mean = _tensor_scalar(out.get("variational_language_inactive_sigma_mean"))
                variational_language_base_topk_score_mean = _tensor_scalar(out.get("variational_language_base_topk_score_mean"))
                variational_language_sampled_topk_score_mean = _tensor_scalar(out.get("variational_language_sampled_topk_score_mean"))
                variational_language_eval_samples = _tensor_scalar(out.get("variational_language_eval_samples"))
                variational_language_prob_var_mean = _tensor_scalar(out.get("variational_language_prob_var_mean"))
                variational_language_prob_std_mean = _tensor_scalar(out.get("variational_language_prob_std_mean"))
                variational_language_predictive_entropy_mean = _tensor_scalar(out.get("variational_language_predictive_entropy_mean"))
                variational_language_logit_var_mean = _tensor_scalar(out.get("variational_language_logit_var_mean"))
                hard_gate_triggered = False
                # Decision logic. Priority order:
                # 0. PresentHead (--use_present_head): present_logit > threshold → present;
                #    else force empty. Takes precedence over all other rules.
                # 1. Inline UCT branch (--use_uncertain_token): U > tau → empty.
                # 2. Plain baseline: pixel_thresh fallback only.
                use_ph = bool(getattr(args, "use_present_head", False))
                if use_ph and present_logit is not None:
                    pl_val = present_logit
                    present_prob = float(torch.sigmoid(torch.tensor(pl_val)).item())
                    head_pred_present = pl_val > present_head_threshold
                    if pl_val <= present_head_threshold:
                        hard_gate_triggered = True
                        pred = torch.zeros_like(pred_raw)
                    else:
                        pred = pred_raw
                    U_val = pl_val   # log present_logit in the "U" column for parity
                elif use_unc and out.get("U") is not None:
                    U_val = float(out["U"].item())
                    if U_val > tau:
                        hard_gate_triggered = True
                        pred = torch.zeros_like(pred_raw)
                    else:
                        pred = pred_raw
                else:
                    pred = pred_raw

                pixels_after_gate = int(pred.sum().item())
                pixel_thresh_triggered = (pixels_after_gate < pixel_thresh) and (not hard_gate_triggered)
                # "pred_empty" semantically: the model said no foreground.
                pred_empty = (pixels_after_gate < pixel_thresh)
                is_neg = bool(view.is_negative[i]) if i < len(view.is_negative) else False
                is_present = not is_neg
                pred_present = not pred_empty
                total += 1

                iou = float("nan")
                if is_neg:
                    n_neg += 1
                    if pred_empty:
                        n_neg_correct += 1
                        giou_sum += 1.0
                        if hard_gate_triggered and pixels_before_gate < pixel_thresh:
                            neg_path_counts["both"] += 1
                        elif hard_gate_triggered:
                            neg_path_counts["hard_gate_only"] += 1
                        else:
                            neg_path_counts["pixel_thresh_only"] += 1
                    else:
                        giou_sum += 0.0
                        neg_path_counts["neither_wrong_nonempty"] += 1
                else:
                    n_pos += 1
                    if not pred_empty:
                        n_pos_correct += 1
                    gt_mask = view.gt_mask[view.category[i]].to("cuda")
                    pred_bool = pred.bool()
                    gt_bool = gt_mask.bool()
                    iou_val = calculate_iou(pred_bool, gt_bool)
                    if np.isnan(iou_val):
                        iou_val = 0.0
                    iou = float(iou_val)
                    giou_sum += iou
                    pos_ious.append(iou)
                    # Why did a positive sample get killed (if it did)?
                    if pred_empty:
                        if hard_gate_triggered and pixels_before_gate < pixel_thresh:
                            pos_killed["both"] += 1
                        elif hard_gate_triggered:
                            pos_killed["hard_gate"] += 1
                        else:
                            pos_killed["pixel_thresh"] += 1
                    else:
                        # Survived, but did it have any overlap with GT?
                        if iou > 0:
                            pos_killed["survived_correct"] += 1
                        else:
                            pos_killed["survived_wrong_iou0"] += 1

                if diagnostic:
                    negative_type = getattr(args, "perturb_variant", "") if is_neg else ""
                    sentence = view.sentence[i] if i < len(view.sentence) else ""
                    category = view.category[i] if i < len(view.category) else ""
                    image_name = getattr(view, "image_name", str(view_idx))
                    query_id = f"{image_name}:{i}"
                    diag_rows.append({
                        "query_id": query_id,
                        "view_idx": view_idx,
                        "sent_idx": i,
                        "image_name": image_name,
                        "sentence": sentence,
                        "category": category,
                        "is_present": int(is_present),
                        "is_negative": int(is_neg),
                        "negative_type": negative_type,
                        "present_logit": present_logit if present_logit is not None else "",
                        "present_prob": present_prob if present_prob is not None else "",
                        "head_pred_present": int(head_pred_present) if head_pred_present is not None else "",
                        "pred_present": int(bool(pred_present)),
                        "pred_empty": int(pred_empty),
                        "mask_iou": iou,
                        "max_score": max_score if max_score is not None else "",
                        "topk_score_mean": topk_score_mean if topk_score_mean is not None else "",
                        "refer_uncertainty_mean": refer_uncertainty_mean if refer_uncertainty_mean is not None else "",
                        "refer_uncertainty_top_mean": refer_uncertainty_top_mean if refer_uncertainty_top_mean is not None else "",
                        "refer_uncertainty_rel_mean": refer_uncertainty_rel_mean if refer_uncertainty_rel_mean is not None else "",
                        "refer_uncertainty_rel_top_mean": refer_uncertainty_rel_top_mean if refer_uncertainty_rel_top_mean is not None else "",
                        "refer_uncertainty_top_std": refer_uncertainty_top_std if refer_uncertainty_top_std is not None else "",
                        "refer_uncertainty_mu_mean": refer_uncertainty_mu_mean if refer_uncertainty_mu_mean is not None else "",
                        "refer_uncertainty_mu_top_mean": refer_uncertainty_mu_top_mean if refer_uncertainty_mu_top_mean is not None else "",
                        "refer_uncertainty_sample_rel_top_mean": refer_uncertainty_sample_rel_top_mean if refer_uncertainty_sample_rel_top_mean is not None else "",
                        "score_sensitivity_top_mean": score_sensitivity_top_mean if score_sensitivity_top_mean is not None else "",
                        "score_reparam_sensitivity_top_mean": score_reparam_sensitivity_top_mean if score_reparam_sensitivity_top_mean is not None else "",
                        "score_target_u_top_mean": score_target_u_top_mean if score_target_u_top_mean is not None else "",
                        "score_u_spearman_top": score_u_spearman_top if score_u_spearman_top is not None else "",
                        "gaussian_attr_conv_topk_count": gaussian_attr_conv_topk_count if gaussian_attr_conv_topk_count is not None else "",
                        "gaussian_attr_conv_pre_adaptive_tokens": gaussian_attr_conv_pre_adaptive_tokens if gaussian_attr_conv_pre_adaptive_tokens is not None else "",
                        "gaussian_attr_conv_pooled_tokens": gaussian_attr_conv_pooled_tokens if gaussian_attr_conv_pooled_tokens is not None else "",
                        "gaussian_attr_conv_uc_topk_mean": gaussian_attr_conv_uc_topk_mean if gaussian_attr_conv_uc_topk_mean is not None else "",
                        "gaussian_attr_conv_uc_topk_std": gaussian_attr_conv_uc_topk_std if gaussian_attr_conv_uc_topk_std is not None else "",
                        "variational_language_kl": variational_language_kl if variational_language_kl is not None else "",
                        "variational_language_mu_norm_mean": variational_language_mu_norm_mean if variational_language_mu_norm_mean is not None else "",
                        "variational_language_base_norm_mean": variational_language_base_norm_mean if variational_language_base_norm_mean is not None else "",
                        "variational_language_posterior_mean_norm_mean": variational_language_posterior_mean_norm_mean if variational_language_posterior_mean_norm_mean is not None else "",
                        "variational_language_sigma_mean": variational_language_sigma_mean if variational_language_sigma_mean is not None else "",
                        "variational_language_log_sigma_mean": variational_language_log_sigma_mean if variational_language_log_sigma_mean is not None else "",
                        "variational_language_mean_shift_norm_mean": variational_language_mean_shift_norm_mean if variational_language_mean_shift_norm_mean is not None else "",
                        "variational_language_active_gate_ratio": variational_language_active_gate_ratio if variational_language_active_gate_ratio is not None else "",
                        "variational_language_active_mu_norm_mean": variational_language_active_mu_norm_mean if variational_language_active_mu_norm_mean is not None else "",
                        "variational_language_inactive_mu_norm_mean": variational_language_inactive_mu_norm_mean if variational_language_inactive_mu_norm_mean is not None else "",
                        "variational_language_active_sigma_mean": variational_language_active_sigma_mean if variational_language_active_sigma_mean is not None else "",
                        "variational_language_inactive_sigma_mean": variational_language_inactive_sigma_mean if variational_language_inactive_sigma_mean is not None else "",
                        "variational_language_base_topk_score_mean": variational_language_base_topk_score_mean if variational_language_base_topk_score_mean is not None else "",
                        "variational_language_sampled_topk_score_mean": variational_language_sampled_topk_score_mean if variational_language_sampled_topk_score_mean is not None else "",
                        "variational_language_eval_samples": variational_language_eval_samples if variational_language_eval_samples is not None else "",
                        "variational_language_prob_var_mean": variational_language_prob_var_mean if variational_language_prob_var_mean is not None else "",
                        "variational_language_prob_std_mean": variational_language_prob_std_mean if variational_language_prob_std_mean is not None else "",
                        "variational_language_predictive_entropy_mean": variational_language_predictive_entropy_mean if variational_language_predictive_entropy_mean is not None else "",
                        "variational_language_logit_var_mean": variational_language_logit_var_mean if variational_language_logit_var_mean is not None else "",
                        "U": U_val if U_val is not None else "",
                        "pixels_before_gate": pixels_before_gate,
                        "pixels_after_gate": pixels_after_gate,
                        "hard_gate_triggered": int(hard_gate_triggered),
                        "pixel_thresh_triggered": int(pixel_thresh_triggered),
                    })
                    if present_prob is not None:
                        diag_labels.append(int(is_present))
                        diag_present_scores.append(present_prob)

    n_acc = (n_neg_correct / n_neg) if n_neg > 0 else float("nan")
    t_acc = (n_pos_correct / n_pos) if n_pos > 0 else float("nan")
    giou = (giou_sum / total) if total > 0 else 0.0
    miou_pos = (sum(pos_ious) / len(pos_ious)) if pos_ious else float("nan")
    p_at_05 = (sum(1 for x in pos_ious if x >= 0.5) / len(pos_ious)) if pos_ious else float("nan")

    print(f"Total samples: {total}  (positive: {n_pos}, negative: {n_neg})")
    print(f"N-acc: {n_acc:.4f}")
    print(f"T-acc: {t_acc:.4f}")
    print(f"gIoU : {giou:.4f}")
    print(f"mIoU+: {miou_pos:.4f}")
    print(f"P@0.5: {p_at_05:.4f}")

    if diagnostic:
        auroc = _binary_auroc(diag_labels, diag_present_scores)
        per_neg_type = defaultdict(lambda: {"total": 0, "false_positive": 0})
        for row in diag_rows:
            if row["is_negative"] != 1:
                continue
            neg_type = row["negative_type"] or "negative"
            per_neg_type[neg_type]["total"] += 1
            if row["pred_present"]:
                per_neg_type[neg_type]["false_positive"] += 1

        per_neg_type_fpr = {}
        for neg_type, counts in sorted(per_neg_type.items()):
            total_neg = counts["total"]
            fp = counts["false_positive"]
            per_neg_type_fpr[neg_type] = {
                "false_positive": fp,
                "total": total_neg,
                "fpr": (fp / total_neg) if total_neg else float("nan"),
            }

        # Summary stats
        U_neg = [r["U"] for r in diag_rows if r["is_negative"] == 1 and r["U"] != ""]
        U_pos = [r["U"] for r in diag_rows if r["is_negative"] == 0 and r["U"] != ""]
        if U_neg:
            print(f"[diag] U/present_logit on negatives: mean={np.mean(U_neg):.4f} "
                  f"median={np.median(U_neg):.4f} "
                  f"min={np.min(U_neg):.4f} max={np.max(U_neg):.4f}")
        if U_pos:
            print(f"[diag] U/present_logit on positives: mean={np.mean(U_pos):.4f} "
                  f"median={np.median(U_pos):.4f} "
                  f"min={np.min(U_pos):.4f} max={np.max(U_pos):.4f}")
        print(f"[diag] positive recall      : {t_acc:.4f}")
        print(f"[diag] negative rejection  : {n_acc:.4f}")
        print(f"[diag] negative FPR        : {1.0 - n_acc:.4f}")
        if not math.isnan(auroc):
            print(f"[diag] AUROC              : {auroc:.4f}")
        else:
            print("[diag] AUROC              : nan")
        for neg_type, stats in per_neg_type_fpr.items():
            print(f"[diag] FPR[{neg_type}]       : {stats['fpr']:.4f} "
                  f"({stats['false_positive']} / {stats['total']})")
        print(f"[diag] Negative path attribution (correct empties):")
        print(f"        hard_gate_only      : {neg_path_counts['hard_gate_only']} / {n_neg}")
        print(f"        pixel_thresh_only   : {neg_path_counts['pixel_thresh_only']} / {n_neg}")
        print(f"        both (redundant)    : {neg_path_counts['both']} / {n_neg}")
        print(f"        wrong (non-empty)   : {neg_path_counts['neither_wrong_nonempty']} / {n_neg}")
        print(f"[diag] Positive sample fates:")
        print(f"        killed by hard_gate : {pos_killed['hard_gate']} / {n_pos}")
        print(f"        killed by pix_thresh: {pos_killed['pixel_thresh']} / {n_pos}")
        print(f"        killed by both      : {pos_killed['both']} / {n_pos}")
        print(f"        survived w/ iou>0   : {pos_killed['survived_correct']} / {n_pos}")
        print(f"        survived w/ iou=0   : {pos_killed['survived_wrong_iou0']} / {n_pos}")

        diag_tag = getattr(args, "diag_tag", "diag")
        csv_path = os.path.join(args.model_path, f"{diag_tag}_diagnostic.csv")
        summary_path = os.path.join(args.model_path, f"{diag_tag}_summary.json")
        if diag_rows:
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=DIAG_FIELDNAMES)
                writer.writeheader()
                writer.writerows(diag_rows)
            print(f"[diag] Wrote per-sample CSV: {csv_path}")
        summary = {
            "total": total,
            "positive": n_pos,
            "negative": n_neg,
            "positive_mIoU": _json_float(miou_pos),
            "positive_recall": _json_float(t_acc),
            "negative_rejection_accuracy": _json_float(n_acc),
            "negative_FPR": _json_float(1.0 - n_acc),
            "AUROC": _json_float(auroc),
            "gIoU": _json_float(giou),
            "P@0.5": _json_float(p_at_05),
            "per_negative_type_FPR": {
                k: {
                    "false_positive": v["false_positive"],
                    "total": v["total"],
                    "fpr": _json_float(v["fpr"]),
                }
                for k, v in per_neg_type_fpr.items()
            },
            "negative_path_counts": neg_path_counts,
            "positive_fates": pos_killed,
        }
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"[diag] Wrote summary JSON: {summary_path}")

    return {"N-acc": n_acc, "T-acc": t_acc, "gIoU": giou,
            "mIoU+": miou_pos, "P@0.5": p_at_05,
            "n_pos": n_pos, "n_neg": n_neg, "total": total}


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="ReferSplat N-acc/T-acc/gIoU evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="chkpnt_cbasetea2514.pth")
    parser.add_argument("--use_uncertain_token", action="store_true")
    # `--unctoken_arch` and `--unctoken_query_mode` come from train.py's
    # OptimizationParams and are saved in cfg_args. Allow them on the eval
    # CLI too so launch scripts that re-pass them (for clarity) don't crash.
    # The values are otherwise picked up from cfg_args via get_combined_args.
    parser.add_argument("--unctoken_arch", type=str, default="external",
                        choices=["external", "inline", "query_concat"])
    parser.add_argument("--unctoken_query_mode", type=str, default="fr_plus_fp",
                        choices=["fr_plus_fp", "frpost_plus_fp", "frpost_only"])
    # Training-only flags that may be re-passed by orchestrator scripts for
    # documentation. Accepted-and-ignored on the eval CLI; eval reads what it
    # needs from cfg_args (or directly from the loaded ckpt's module shapes).
    parser.add_argument("--use_kendall_self", action="store_true",
                        help="(training-only, accepted-and-ignored at eval)")
    parser.add_argument("--use_kendall_aux", action="store_true",
                        help="(training-only, accepted-and-ignored at eval)")
    parser.add_argument("--uncertain_tau", type=float, default=0.5)
    parser.add_argument("--uncertain_gamma", type=float, default=1.0)
    parser.add_argument("--pixel_thresh", type=int, default=50,
                        help="Pred is treated as empty if pred.sum() < this many positive pixels.")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Emit per-sample U / pixel counts / path attribution and CSV.")
    parser.add_argument("--diag_tag", type=str, default="diag",
                        help="Tag prefix for the diagnostic CSV file name.")
    parser.add_argument("--use_present_head", action="store_true",
                        help="Use the PresentHead FFN classifier for present/absent "
                             "decision (highest priority over U).")
    parser.add_argument("--use_q_nt", action="store_true",
                        help="Use q_nt no-target query token(s) for the PresentHead input.")
    parser.add_argument("--q_nt_num_queries", type=int, default=1,
                        help="Number of q_nt query tokens. Use 4 with --q_nt_pool gap for q_nt4 GAP.")
    parser.add_argument("--q_nt_pool", type=str, default="first",
                        choices=["first", "gap"],
                        help="How to reduce q_nt outputs before PresentHead.")
    parser.add_argument("--q_nt_no_fp", action="store_true",
                        help="Use a pure q_nt query without learnable pseudo-position f_p_nt.")
    parser.add_argument("--use_scene_aware_fusion", action="store_true",
                        help="Feed PresentHead concat(query_feature, pooled_g, max_score, topk_score_mean).")
    parser.add_argument("--use_topk_evidence_gap", action="store_true",
                        help="Feed PresentHead GAP(g_topk), where top-k is selected by text-response score.")
    parser.add_argument("--use_topk_evidence_fusion", action="store_true",
                        help="Feed PresentHead concat(query_feature, GAP(g_topk)).")
    parser.add_argument("--topk_evidence_ratio", type=float, default=-1.0,
                        help="Fraction of Gaussians used for top-k evidence pooling. "
                             "<0 reuses render()'s contrastive top-k ratio.")
    parser.add_argument("--fusion_layer_norm", action="store_true",
                        help="Apply non-parametric LayerNorm to fusion vectors before PresentHead.")
    parser.add_argument("--fusion_query_layer_norm", action="store_true",
                        help="Apply non-parametric LayerNorm only to query_feature before top-k fusion.")
    parser.add_argument("--fusion_detach_pooled_g", action="store_true",
                        help="Detach pooled_g before PresentHead so classifier gradients do not update g through fusion.")
    parser.add_argument("--use_bald_evidence_weight", action="store_true",
                        help="Use probe-view BALD to weight top-k fusion evidence pooling.")
    parser.add_argument("--bald_weight_mode", type=str, default="stable",
                        choices=["stable", "uncertain"],
                        help="'stable' pools with 1-BALD; 'uncertain' pools with BALD.")
    parser.add_argument("--bald_weight_eps", type=float, default=0.05)
    parser.add_argument("--bald_probe_max_angle", type=float, default=60.0)
    parser.add_argument("--bald_probe_strategy", type=str, default="nearest",
                        choices=["random", "nearest", "farthest"],
                        help="Probe view selection for BALD at eval/test time.")
    parser.add_argument("--test_neg_target_ratio", type=float, default=0.15,
                        help="If in [0,1], stratified-subsample the eval set "
                             "so neg/(pos+neg) ≈ this target. Default 0.15 "
                             "matches the current training prior; -1 = use native "
                             "ratio. "
                             "Useful for aligning eval prior with training "
                             "neg ratio (--training_neg_target_ratio).")
    parser.add_argument("--test_seed", type=int, default=-1,
                        help="Seed for pos/neg subsampling under "
                             "--test_neg_target_ratio. -1 (default) = truly "
                             "random per run (os.urandom). Set to a fixed int "
                             "for reproducible eval subsets.")
    parser.add_argument("--present_head_hidden", type=int, default=128)
    parser.add_argument("--present_head_dropout", type=float, default=0.1)
    parser.add_argument("--present_head_eps", type=float, default=0.05)
    parser.add_argument("--present_head_threshold", type=float, default=0.0,
                        help="PresentHead logit threshold; present iff logit > threshold.")
    parser.add_argument("--use_gaussian_attr_conv_head", action="store_true",
                        help="Use Gaussian attribute + external UC Conv1d pooling head before PresentHead.")
    parser.add_argument("--external_gaussian_uncertainty_path", type=str, default="")
    parser.add_argument("--external_gaussian_uncertainty_key", type=str, default="")
    parser.add_argument("--gaussian_attr_conv_pooled_tokens", type=int, default=64)
    parser.add_argument("--gaussian_attr_conv_num_layers", type=int, default=2)
    parser.add_argument("--gaussian_attr_conv_num_heads", type=int, default=4)
    parser.add_argument("--gaussian_attr_conv_ffn_dim", type=int, default=256)
    parser.add_argument("--gaussian_attr_conv_dropout", type=float, default=0.0)
    parser.add_argument("--gaussian_attr_conv_kernel_size", type=int, default=5)
    parser.add_argument("--use_refer_uncertainty", action="store_true")
    parser.add_argument("--refer_uncertainty_only", action="store_true")
    parser.add_argument("--refer_uncertainty_score_only", action="store_true")
    parser.add_argument("--refer_uncertainty_prior_std", type=float, default=0.0025)
    parser.add_argument("--refer_uncertainty_kl_weight", type=float, default=1e-4)
    parser.add_argument("--refer_uncertainty_score_topk_ratio", type=float, default=0.03)
    parser.add_argument("--refer_uncertainty_score_probe_std", type=float, default=0.0025)
    parser.add_argument("--refer_uncertainty_score_target_scale", type=float, default=0.5)
    parser.add_argument("--refer_uncertainty_score_loss_weight", type=float, default=1.0)
    parser.add_argument("--refer_uncertainty_reparam_score_weight", type=float, default=1.0)
    parser.add_argument("--refer_uncertainty_log_sigma_min", type=float, default=-5.0)
    parser.add_argument("--refer_uncertainty_log_sigma_max", type=float, default=2.0)
    parser.add_argument("--use_variational_language", action="store_true")
    parser.add_argument("--variational_language_prior_std", type=float, default=0.0025)
    parser.add_argument("--variational_language_kl_weight", type=float, default=1e-4)
    parser.add_argument("--variational_language_kl_warmup_iters", type=int, default=2000)
    parser.add_argument("--variational_language_gate_warmup_iters", type=int, default=2000)
    parser.add_argument("--variational_language_offset_lr_scale", type=float, default=0.1)
    parser.add_argument("--variational_language_mu_lr", type=float, default=0.0)
    parser.add_argument("--variational_language_sigma_lr", type=float, default=0.0)
    parser.add_argument("--variational_language_log_sigma_min", type=float, default=-5.0)
    parser.add_argument("--variational_language_log_sigma_max", type=float, default=2.0)
    parser.add_argument("--variational_language_eval_samples", type=int, default=10)
    parser.add_argument("--reset_iter_on_restore", action="store_true")
    # --perturb_variant is auto-registered by ModelParams (default "").
    # Set to one of {"attribute","category","spatial"} for zero-shot eval
    # on the perturbation variants.
    args = get_combined_args(parser)
    args.include_feature = True
    if args.iteration == -1:
        m = re.search(r"chkpnt_cbasetea251(\d+)\.pth$", args.checkpoint_name)
        args.iteration = int(m.group(1)) if m else 0
    safe_state(args.quiet)
    evaluate(model.extract(args), pipeline.extract(args), args)
