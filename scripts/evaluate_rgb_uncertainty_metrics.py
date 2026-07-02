import argparse
import csv
import json
import math
import os
import sys
from os import makedirs

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import GaussianModel, rasterize_per_gaussian_scalar
from scene import Scene
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh


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


def rgb_render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=True,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        scales = None
        rotations = None
    else:
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        shs = None
    else:
        colors_precomp = None
        shs = pc.get_features

    dummy_feature = torch.zeros(
        (pc.get_xyz.shape[0], 1), dtype=pc.get_xyz.dtype, device=pc.get_xyz.device
    )
    rendered_image, _, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=dummy_feature,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return rendered_image.clamp(0.0, 1.0)


def rankdata_average(values):
    values = values.astype(np.float64, copy=False)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    sorted_values = values[order]
    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    starts = np.flatnonzero(group_start)
    ends = np.r_[starts[1:], n]
    avg_ranks = (starts.astype(np.float64) + 1.0 + ends.astype(np.float64)) / 2.0
    ranks[order] = np.repeat(avg_ranks, ends - starts)
    return ranks


def spearman_corr(values, scores):
    values = values.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    ok = np.isfinite(values) & np.isfinite(scores)
    values = values[ok]
    scores = scores[ok]
    if values.size <= 1:
        return float("nan")
    rv = rankdata_average(values)
    rs = rankdata_average(scores)
    rv -= rv.mean()
    rs -= rs.mean()
    denom = np.linalg.norm(rv) * np.linalg.norm(rs)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(rv, rs) / denom)


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


def psnr_from_mse(mse):
    mse = float(mse)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def rgb_risk_coverage(errors, scores, num_points=101):
    errors = errors.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    ok = np.isfinite(errors) & np.isfinite(scores)
    errors = errors[ok]
    scores = scores[ok]
    n = errors.size
    if n == 0:
        return [], float("nan"), float("nan"), float("nan")

    order_model = np.argsort(scores, kind="mergesort")
    errors_by_certainty = errors[order_model]
    cumsum_model = np.cumsum(errors_by_certainty, dtype=np.float64)

    order_oracle = np.argsort(errors, kind="mergesort")
    errors_by_oracle = errors[order_oracle]
    cumsum_oracle = np.cumsum(errors_by_oracle, dtype=np.float64)

    coverages = np.linspace(0.01, 1.0, int(num_points), dtype=np.float64)
    rows = []
    risks = []
    oracle_risks = []
    for coverage in coverages:
        k = int(math.ceil(float(coverage) * n))
        k = min(max(k, 1), n)
        risk = float(cumsum_model[k - 1] / float(k))
        oracle_risk = float(cumsum_oracle[k - 1] / float(k))
        rows.append({
            "coverage": float(coverage),
            "removed_fraction": float(1.0 - coverage),
            "mse": risk,
            "psnr": psnr_from_mse(risk),
            "oracle_mse": oracle_risk,
            "oracle_psnr": psnr_from_mse(oracle_risk),
            "sparsification_error": float(risk - oracle_risk),
        })
        risks.append(risk)
        oracle_risks.append(oracle_risk)

    risks = np.asarray(risks, dtype=np.float64)
    oracle_risks = np.asarray(oracle_risks, dtype=np.float64)
    ause = float(np.trapz(risks - oracle_risks, coverages))
    aurc = float(np.trapz(risks, coverages))
    full_mse = float(errors.mean())
    rel_ause = float(ause / max(full_mse, 1e-12))
    return rows, ause, rel_ause, aurc


def maybe_subsample(errors, scores, max_pixels, seed):
    if max_pixels <= 0 or errors.size <= max_pixels:
        return errors, scores
    rng = np.random.default_rng(seed)
    idx = rng.choice(errors.size, size=max_pixels, replace=False)
    return errors[idx], scores[idx]


def compute_metrics(errors, scores, high_error_quantile, curve_points):
    threshold = float(np.quantile(errors, float(high_error_quantile)))
    labels = (errors >= threshold).astype(np.uint8)
    curve, ause, rel_ause, aurc = rgb_risk_coverage(errors, scores, curve_points)
    out = {
        "num_pixels": int(errors.size),
        "mse": float(errors.mean()),
        "rmse": float(np.sqrt(errors.mean())),
        "psnr": psnr_from_mse(float(errors.mean())),
        "mae_rgb_mse_sqrt_proxy": float(np.mean(np.sqrt(np.maximum(errors, 0.0)))),
        "high_error_quantile": float(high_error_quantile),
        "high_error_threshold": threshold,
        "high_error_rate": float(labels.mean()),
        "high_error_auroc": binary_auroc(labels, scores),
        "high_error_auprc": average_precision(labels, scores),
        "spearman_error_uncertainty": spearman_corr(errors, scores),
        "ause_mse": ause,
        "ause_mse_relative": rel_ause,
        "aurc_mse": aurc,
    }
    for target in (0.90, 0.75, 0.50):
        closest = min(curve, key=lambda r: abs(r["coverage"] - target)) if curve else None
        if closest is not None:
            out[f"mse_at_coverage_{int(target * 100)}"] = closest["mse"]
            out[f"psnr_at_coverage_{int(target * 100)}"] = closest["psnr"]
    return out, curve


def evaluate(dataset, pipeline, args):
    if not args.checkpoint_name:
        raise ValueError("Pass the explicit best checkpoint via --checkpoint_name.")
    if not args.uncertainty_path:
        raise ValueError("--uncertainty_path is required.")

    gaussians = build_gaussians(dataset, args)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint = os.path.join(args.model_path, args.checkpoint_name)
    model_params, first_iter = torch.load(
        checkpoint, map_location=f"cuda:{torch.cuda.current_device()}", weights_only=False
    )
    gaussians.restore(model_params, args, mode="test")

    uc = torch.load(args.uncertainty_path, map_location="cpu", weights_only=False)
    if args.uncertainty_key not in uc:
        raise KeyError(f"{args.uncertainty_key} not found in {args.uncertainty_path}")
    scalar = uc[args.uncertainty_key].float()
    if scalar.numel() != gaussians.get_xyz.shape[0]:
        raise ValueError(
            f"Uncertainty count {scalar.numel()} does not match Gaussian count {gaussians.get_xyz.shape[0]}"
        )
    scalar = scalar.cuda(non_blocking=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    feature_background = torch.zeros_like(background)

    errors_all = []
    scores_all = []
    image_rows = []
    views = scene.getTestCameras()
    if int(getattr(args, "max_test_views", -1)) > 0:
        views = views[: int(args.max_test_views)]

    with torch.no_grad():
        for view_idx, view in enumerate(tqdm(views, desc="RGB uncertainty eval")):
            pred = rgb_render(view, gaussians, pipeline, background)
            gt = view.original_image[0:3, :, :].to("cuda")
            err = (pred - gt).pow(2).mean(dim=0)
            score = rasterize_per_gaussian_scalar(
                scalar, view, gaussians, pipeline, feature_background,
            )
            if score.dim() == 3:
                score = score.squeeze(0)
            errors_all.append(err.detach().cpu().numpy().reshape(-1).astype(np.float32))
            scores_all.append(score.detach().cpu().numpy().reshape(-1).astype(np.float32))
            image_rows.append({
                "view_idx": view_idx,
                "image_name": getattr(view, "image_name", str(view_idx)),
                "mse": float(err.mean().item()),
                "psnr": psnr_from_mse(float(err.mean().item())),
                "uncertainty_mean": float(score.mean().item()),
                "uncertainty_min": float(score.min().item()),
                "uncertainty_max": float(score.max().item()),
            })

    errors_all = np.concatenate(errors_all)
    scores_all = np.concatenate(scores_all)
    errors_eval, scores_eval = maybe_subsample(
        errors_all, scores_all, int(args.max_metric_pixels), int(args.metric_seed)
    )
    metrics, curve = compute_metrics(
        errors_eval, scores_eval, float(args.high_error_quantile), int(args.curve_points)
    )

    output_dir = args.output_dir or os.path.join(
        args.model_path,
        "rgb_uncertainty_error_metrics",
        f"{os.path.splitext(args.checkpoint_name)[0]}_{args.uncertainty_key}",
    )
    makedirs(output_dir, exist_ok=True)

    summary = {
        "source_path": dataset.source_path,
        "model_path": args.model_path,
        "checkpoint": checkpoint,
        "first_iter_in_checkpoint": int(first_iter) if isinstance(first_iter, int) else str(first_iter),
        "uncertainty_path": args.uncertainty_path,
        "uncertainty_key": args.uncertainty_key,
        "num_test_views": len(image_rows),
        "max_metric_pixels": int(args.max_metric_pixels),
        "metrics_rgb_pixels": metrics,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "per_image.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    with open(os.path.join(output_dir, "risk_coverage_rgb_mse.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "coverage", "removed_fraction", "mse", "psnr",
                "oracle_mse", "oracle_psnr", "sparsification_error",
            ],
        )
        writer.writeheader()
        writer.writerows(curve)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-Gaussian uncertainty against RGB render error.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--uncertainty_path", type=str, required=True)
    parser.add_argument("--uncertainty_key", type=str, default="pup_uncertainty_rank01")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--curve_points", type=int, default=101)
    parser.add_argument("--high_error_quantile", type=float, default=0.90)
    parser.add_argument("--max_test_views", type=int, default=-1)
    parser.add_argument("--max_metric_pixels", type=int, default=0,
                        help="0 = exact; otherwise deterministic subsample for rank metrics.")
    parser.add_argument("--metric_seed", type=int, default=20260628)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = get_combined_args(parser)
    args.include_feature = True
    safe_state(args.quiet)
    evaluate(model.extract(args), pipeline.extract(args), args)


if __name__ == "__main__":
    main()
