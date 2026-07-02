import argparse
import csv
import json
import math
import os
import random
import sys
from os import makedirs

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import (
    GaussianModel,
    rasterize_per_gaussian_scalar,
    render_variational_language_mc,
)
from scene import Scene
from test_miou import calculate_iou
from utils.general_utils import safe_state


def build_gaussians(dataset, args):
    return GaussianModel(
        dataset.sh_degree,
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
    )


def binary_auroc(labels, scores):
    labels = labels.astype(np.uint8, copy=False)
    scores = scores.astype(np.float64, copy=False)
    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]
    n = labels.size
    if n == 0:
        return float("nan")
    n_pos = int(labels.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata_average(scores)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels, scores):
    labels = labels.astype(np.uint8, copy=False)
    scores = scores.astype(np.float64, copy=False)
    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.float64)
    precision = tp / (np.arange(sorted_labels.size, dtype=np.float64) + 1.0)
    return float((precision * sorted_labels).sum() / float(n_pos))


def rankdata_average(values):
    values = values.astype(np.float64, copy=False)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_values = values[order]
    if n == 0:
        return ranks
    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    starts = np.flatnonzero(group_start)
    ends = np.r_[starts[1:], n]
    avg_ranks = (starts.astype(np.float64) + 1.0 + ends.astype(np.float64)) / 2.0
    ranks[order] = np.repeat(avg_ranks, ends - starts)
    return ranks


def spearman_corr(labels, scores):
    labels = labels.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]
    if labels.size <= 1 or labels.min() == labels.max():
        return float("nan")
    rs = rankdata_average(scores)
    n = labels.size
    n0 = float((labels == 0).sum())
    n1 = float(n - n0)
    rank0 = (1.0 + n0) / 2.0
    rank1 = (n0 + 1.0 + n) / 2.0
    rl = np.where(labels == 1, rank1, rank0).astype(np.float64, copy=False)
    rs -= rs.mean()
    rl -= rl.mean()
    denom = np.linalg.norm(rs) * np.linalg.norm(rl)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(rs, rl) / denom)


def risk_coverage(labels, scores, num_points=101):
    labels = labels.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]
    n = labels.size
    if n == 0:
        return [], float("nan"), float("nan")

    order_model = np.argsort(scores, kind="mergesort")
    errors_by_certainty = labels[order_model]
    cum_errors = np.cumsum(errors_by_certainty, dtype=np.float64)

    total_errors = float(labels.sum())
    total_correct = float(n - total_errors)
    coverages = np.linspace(0.01, 1.0, int(num_points), dtype=np.float64)
    rows = []
    risks = []
    oracle_risks = []
    for coverage in coverages:
        k = int(math.ceil(float(coverage) * n))
        k = min(max(k, 1), n)
        risk = float(cum_errors[k - 1] / float(k))
        oracle_errors = max(0.0, float(k) - total_correct)
        oracle_risk = float(oracle_errors / float(k))
        rows.append({
            "coverage": float(coverage),
            "removed_fraction": float(1.0 - coverage),
            "risk": risk,
            "pixel_accuracy": float(1.0 - risk),
            "oracle_risk": oracle_risk,
            "sparsification_error": float(risk - oracle_risk),
        })
        risks.append(risk)
        oracle_risks.append(oracle_risk)

    risks = np.asarray(risks, dtype=np.float64)
    oracle_risks = np.asarray(oracle_risks, dtype=np.float64)
    ause = float(np.trapz(risks - oracle_risks, coverages))
    aurc = float(np.trapz(risks, coverages))
    return rows, ause, aurc


def pixel_metrics(labels, scores, num_curve_points=101):
    labels = labels.astype(np.uint8, copy=False)
    scores = scores.astype(np.float32, copy=False)
    rows, ause, aurc = risk_coverage(labels, scores, num_curve_points)

    out = {
        "num_pixels": int(labels.size),
        "num_error_pixels": int(labels.sum()),
        "error_rate": float(labels.mean()) if labels.size else float("nan"),
        "error_auroc": binary_auroc(labels, scores),
        "error_auprc": average_precision(labels, scores),
        "spearman_error_uncertainty": spearman_corr(labels, scores),
        "ause": ause,
        "aurc": aurc,
    }
    for target in (0.90, 0.75, 0.50):
        closest = min(rows, key=lambda r: abs(r["coverage"] - target)) if rows else None
        if closest is not None:
            out[f"risk_at_coverage_{int(target * 100)}"] = closest["risk"]
            out[f"pixel_acc_at_coverage_{int(target * 100)}"] = closest["pixel_accuracy"]
    return out, rows


def maybe_subsample(labels, scores, max_pixels, seed):
    if max_pixels <= 0 or labels.size <= max_pixels:
        return labels, scores
    rng = np.random.default_rng(seed)
    idx = rng.choice(labels.size, size=max_pixels, replace=False)
    return labels[idx], scores[idx]


def render_prediction(view, sentence_idx, gaussians, pipeline, background, args):
    out = render_variational_language_mc(
        view, gaussians, pipeline, background, args,
        sentence=view.sentence[sentence_idx],
    )
    prob = torch.sigmoid(out["language_feature_image"])
    pred = (prob >= 0.5)

    present_logit = out.get("present_logit")
    if getattr(args, "use_present_head", False) and present_logit is not None:
        threshold = float(getattr(args, "present_head_threshold", 0.0))
        if float(present_logit.item()) <= threshold:
            pred = torch.zeros_like(pred, dtype=torch.bool)
    elif getattr(args, "use_uncertain_token", False) and out.get("U") is not None:
        tau = float(getattr(args, "uncertain_tau", 0.5))
        if float(out["U"].item()) > tau:
            pred = torch.zeros_like(pred, dtype=torch.bool)

    if bool(getattr(args, "apply_pixel_thresh_empty", True)):
        pixel_thresh = int(getattr(args, "pixel_thresh", 50))
        if int(pred.sum().item()) < pixel_thresh:
            pred = torch.zeros_like(pred, dtype=torch.bool)

    return pred, prob


def evaluate(dataset, pipeline, args):
    if not args.checkpoint_name:
        raise ValueError("Use the project protocol: pass the explicit best checkpoint via --checkpoint_name.")
    if not args.pup_uncertainty_path:
        raise ValueError("--pup_uncertainty_path is required.")

    gaussians = build_gaussians(dataset, args)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint = os.path.join(args.model_path, args.checkpoint_name)
    model_params, first_iter = torch.load(checkpoint, map_location=f"cuda:{torch.cuda.current_device()}", weights_only=False)
    gaussians.restore(model_params, args, mode="test")

    pup = torch.load(args.pup_uncertainty_path, map_location="cpu", weights_only=False)
    if args.uncertainty_key not in pup:
        raise KeyError(f"{args.uncertainty_key} not found in {args.pup_uncertainty_path}")
    scalar = pup[args.uncertainty_key].float()
    if scalar.numel() != gaussians.get_xyz.shape[0]:
        raise ValueError(
            f"PUP score count {scalar.numel()} does not match Gaussian count {gaussians.get_xyz.shape[0]}"
        )
    scalar = scalar.cuda(non_blocking=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    feature_background = torch.zeros_like(background)
    views = scene.getTestCameras()

    labels_all = []
    scores_all = []
    labels_union = []
    scores_union = []
    sample_rows = []

    with torch.no_grad():
        for view_idx, view in enumerate(tqdm(views, desc="PUP uncertainty eval")):
            unc_map = rasterize_per_gaussian_scalar(
                scalar, view, gaussians, pipeline, feature_background,
            )
            if unc_map.dim() == 2:
                unc_map = unc_map.unsqueeze(0)
            unc_map = unc_map.float()

            for sent_idx in range(len(view.sentence)):
                is_neg = bool(view.is_negative[sent_idx]) if sent_idx < len(view.is_negative) else False
                if is_neg and not bool(getattr(args, "include_negatives", False)):
                    continue

                pred, prob = render_prediction(view, sent_idx, gaussians, pipeline, background, args)
                if is_neg:
                    gt = torch.zeros_like(pred, dtype=torch.bool)
                else:
                    gt = view.gt_mask[view.category[sent_idx]].to("cuda").bool()
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)

                err = (pred.bool() != gt.bool()).squeeze(0)
                score = unc_map.squeeze(0)
                labels_all.append(err.detach().cpu().numpy().reshape(-1).astype(np.uint8))
                scores_all.append(score.detach().cpu().numpy().reshape(-1).astype(np.float32))

                union = (pred.bool() | gt.bool()).squeeze(0)
                if bool(union.any().item()):
                    labels_union.append(err[union].detach().cpu().numpy().reshape(-1).astype(np.uint8))
                    scores_union.append(score[union].detach().cpu().numpy().reshape(-1).astype(np.float32))

                pred_bool = pred.bool()
                gt_bool = gt.bool()
                iou = 1.0 if is_neg and not bool(pred_bool.any().item()) else calculate_iou(pred_bool, gt_bool)
                if np.isnan(iou):
                    iou = 0.0
                error_rate = float(err.float().mean().item())
                union_error_rate = float(err[union].float().mean().item()) if bool(union.any().item()) else float("nan")
                sample_rows.append({
                    "view_idx": view_idx,
                    "sent_idx": sent_idx,
                    "image_name": getattr(view, "image_name", str(view_idx)),
                    "category": view.category[sent_idx] if sent_idx < len(view.category) else "",
                    "is_negative": int(is_neg),
                    "iou": float(iou),
                    "pixel_error_rate_all": error_rate,
                    "pixel_error_rate_union": union_error_rate,
                    "uncertainty_mean_all": float(score.mean().item()),
                    "uncertainty_mean_error": float(score[err].mean().item()) if bool(err.any().item()) else float("nan"),
                    "uncertainty_mean_correct": float(score[~err].mean().item()) if bool((~err).any().item()) else float("nan"),
                })

    if not labels_all:
        raise RuntimeError("No evaluation samples were collected.")

    labels_all = np.concatenate(labels_all)
    scores_all = np.concatenate(scores_all)
    labels_union = np.concatenate(labels_union) if labels_union else np.zeros((0,), dtype=np.uint8)
    scores_union = np.concatenate(scores_union) if scores_union else np.zeros((0,), dtype=np.float32)

    labels_all_eval, scores_all_eval = maybe_subsample(
        labels_all, scores_all, int(args.max_metric_pixels), int(args.metric_seed),
    )
    labels_union_eval, scores_union_eval = maybe_subsample(
        labels_union, scores_union, int(args.max_metric_pixels), int(args.metric_seed) + 17,
    )

    metrics_all, curve_all = pixel_metrics(labels_all_eval, scores_all_eval, int(args.curve_points))
    metrics_union, curve_union = pixel_metrics(labels_union_eval, scores_union_eval, int(args.curve_points))

    output_dir = args.output_dir or os.path.join(
        args.model_path,
        "pup_gaussian_uncertainty",
        f"uncertainty_error_metrics_{os.path.splitext(args.checkpoint_name)[0]}_{args.uncertainty_key}",
    )
    makedirs(output_dir, exist_ok=True)

    summary = {
        "source_path": dataset.source_path,
        "model_path": args.model_path,
        "checkpoint": checkpoint,
        "first_iter_in_checkpoint": int(first_iter) if isinstance(first_iter, int) else str(first_iter),
        "pup_uncertainty_path": args.pup_uncertainty_path,
        "uncertainty_key": args.uncertainty_key,
        "include_negatives": bool(args.include_negatives),
        "apply_pixel_thresh_empty": bool(args.apply_pixel_thresh_empty),
        "pixel_thresh": int(args.pixel_thresh),
        "num_samples": len(sample_rows),
        "max_metric_pixels": int(args.max_metric_pixels),
        "metrics_all_pixels": metrics_all,
        "metrics_foreground_union_pixels": metrics_union,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "per_sample.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    for name, rows in (("risk_coverage_all_pixels.csv", curve_all),
                       ("risk_coverage_foreground_union_pixels.csv", curve_union)):
        with open(os.path.join(output_dir, name), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["coverage", "removed_fraction", "risk",
                            "pixel_accuracy", "oracle_risk", "sparsification_error"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_dir}")
    return summary


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="Evaluate PUP per-Gaussian uncertainty against ReferSplat mask errors.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--pup_uncertainty_path", type=str, required=True)
    parser.add_argument("--uncertainty_key", type=str, default="pup_uncertainty_rank01")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--include_negatives", action="store_true", default=False)
    parser.add_argument("--pixel_thresh", type=int, default=50)
    parser.add_argument("--no_pixel_thresh_empty", dest="apply_pixel_thresh_empty", action="store_false")
    parser.set_defaults(apply_pixel_thresh_empty=True)
    parser.add_argument("--curve_points", type=int, default=101)
    parser.add_argument("--max_metric_pixels", type=int, default=0,
                        help="0 = exact; otherwise deterministic subsample for rank metrics.")
    parser.add_argument("--metric_seed", type=int, default=20260627)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = get_combined_args(parser)
    args.include_feature = True
    safe_state(args.quiet)
    evaluate(model.extract(args), pipeline.extract(args), args)


if __name__ == "__main__":
    main()
