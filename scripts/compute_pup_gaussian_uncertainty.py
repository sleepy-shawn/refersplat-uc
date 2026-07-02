import json
import math
import os
import sys
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from uncertainty.fisher import rank01_high
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh

from rasterization_and_pup_fisher import GaussianRasterizationSettings, GaussianRasterizer


def fisher_render(viewpoint_camera, pc, pipe, bg_color, fishers, scaling_modifier=1.0):
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

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
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        fishers=fishers,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def pool_fisher_cuda(view_idx, view, gaussians, pipeline, background, fishers, resolution):
    sym_fishers = torch.zeros(
        (fishers.shape[0], 21), dtype=fishers.dtype, device=fishers.device
    )
    sym_fishers.requires_grad_(True)

    view_copy = view
    original_height = view_copy.image_height
    original_width = view_copy.image_width
    view_copy.image_height = math.ceil(view.image_height / resolution)
    view_copy.image_width = math.ceil(view.image_width / resolution)
    try:
        image = fisher_render(view_copy, gaussians, pipeline, background, sym_fishers)["render"]
        image.sum().backward()
    finally:
        view_copy.image_height = original_height
        view_copy.image_width = original_width

    grad = sym_fishers.grad
    fishers[:, 0, 0] += grad[:, 0]
    fishers[:, 0, 1] += grad[:, 1]
    fishers[:, 0, 2] += grad[:, 2]
    fishers[:, 0, 3] += grad[:, 3]
    fishers[:, 0, 4] += grad[:, 4]
    fishers[:, 0, 5] += grad[:, 5]

    fishers[:, 1, 0] += grad[:, 1]
    fishers[:, 1, 1] += grad[:, 6]
    fishers[:, 1, 2] += grad[:, 7]
    fishers[:, 1, 3] += grad[:, 8]
    fishers[:, 1, 4] += grad[:, 9]
    fishers[:, 1, 5] += grad[:, 10]

    fishers[:, 2, 0] += grad[:, 2]
    fishers[:, 2, 1] += grad[:, 7]
    fishers[:, 2, 2] += grad[:, 11]
    fishers[:, 2, 3] += grad[:, 12]
    fishers[:, 2, 4] += grad[:, 13]
    fishers[:, 2, 5] += grad[:, 14]

    fishers[:, 3, 0] += grad[:, 3]
    fishers[:, 3, 1] += grad[:, 8]
    fishers[:, 3, 2] += grad[:, 12]
    fishers[:, 3, 3] += grad[:, 15]
    fishers[:, 3, 4] += grad[:, 16]
    fishers[:, 3, 5] += grad[:, 17]

    fishers[:, 4, 0] += grad[:, 4]
    fishers[:, 4, 1] += grad[:, 9]
    fishers[:, 4, 2] += grad[:, 13]
    fishers[:, 4, 3] += grad[:, 16]
    fishers[:, 4, 4] += grad[:, 18]
    fishers[:, 4, 5] += grad[:, 19]

    fishers[:, 5, 0] += grad[:, 5]
    fishers[:, 5, 1] += grad[:, 10]
    fishers[:, 5, 2] += grad[:, 14]
    fishers[:, 5, 3] += grad[:, 17]
    fishers[:, 5, 4] += grad[:, 19]
    fishers[:, 5, 5] += grad[:, 20]


def resolve_checkpoint(model_path, checkpoint_name, checkpoint_path):
    if checkpoint_path:
        return checkpoint_path
    if not checkpoint_name:
        raise ValueError(
            "No checkpoint specified. Project protocol: use the best model checkpoint explicitly "
            "with --checkpoint_name or --checkpoint_path; do not silently fall back to final."
        )
    return os.path.join(model_path, checkpoint_name)


def build_gaussians(dataset, opt):
    return GaussianModel(
        dataset.sh_degree,
        use_uncertain_token=getattr(opt, "use_uncertain_token", False),
        unctoken_query_mode=getattr(opt, "unctoken_query_mode", "fr_plus_fp"),
        unctoken_arch=getattr(opt, "unctoken_arch", "external"),
        use_present_head=getattr(opt, "use_present_head", False),
        present_head_hidden=getattr(opt, "present_head_hidden", 128),
        present_head_dropout=getattr(opt, "present_head_dropout", 0.1),
        use_kendall_self=getattr(opt, "use_kendall_self", False),
        use_q_nt=getattr(opt, "use_q_nt", False),
        q_nt_num_queries=getattr(opt, "q_nt_num_queries", 1),
        q_nt_pool=getattr(opt, "q_nt_pool", "first"),
        q_nt_no_fp=getattr(opt, "q_nt_no_fp", False),
        use_scene_aware_fusion=getattr(opt, "use_scene_aware_fusion", False),
        use_topk_evidence_gap=getattr(opt, "use_topk_evidence_gap", False),
        use_topk_evidence_fusion=getattr(opt, "use_topk_evidence_fusion", False),
        fusion_layer_norm=getattr(opt, "fusion_layer_norm", False),
        fusion_query_layer_norm=getattr(opt, "fusion_query_layer_norm", False),
        fusion_detach_pooled_g=getattr(opt, "fusion_detach_pooled_g", False),
        use_bald_evidence_weight=getattr(opt, "use_bald_evidence_weight", False),
        use_refer_uncertainty=getattr(opt, "use_refer_uncertainty", False),
        use_variational_language=getattr(opt, "use_variational_language", False),
        variational_language_prior_std=getattr(opt, "variational_language_prior_std", 0.0025),
        variational_language_log_sigma_min=getattr(opt, "variational_language_log_sigma_min", -5.0),
        variational_language_log_sigma_max=getattr(opt, "variational_language_log_sigma_max", 2.0),
    )


def main():
    parser = ArgumentParser(description="Compute PUP Fisher-based per-Gaussian uncertainty for ReferSplat checkpoints.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--pool_resolution", type=int, default=4)
    parser.add_argument("--max_train_cameras", type=int, default=-1)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--svd_eps", type=float, default=1e-12)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    opt.include_feature = bool(getattr(opt, "include_feature", False))

    gaussians = build_gaussians(dataset, opt)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)

    checkpoint = resolve_checkpoint(dataset.model_path, args.checkpoint_name, args.checkpoint_path)
    model_params, first_iter = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, opt)
    gaussians.eval = lambda: None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = scene.getTrainCameras()
    if args.camera_stride > 1:
        cameras = cameras[:: args.camera_stride]
    if args.max_train_cameras > 0:
        cameras = cameras[: args.max_train_cameras]

    n_gaussians = gaussians.get_xyz.shape[0]
    fishers = torch.zeros(n_gaussians, 6, 6, device="cuda", dtype=torch.float32)

    with torch.enable_grad():
        for view_idx, view in tqdm(
            list(enumerate(cameras)),
            desc=f"PUP Fisher {os.path.basename(checkpoint)}",
        ):
            pool_fisher_cuda(view_idx, view, gaussians, pipe, background, fishers, args.pool_resolution)

    fishers = 0.5 * (fishers + fishers.transpose(1, 2))
    sv = torch.linalg.svdvals(fishers)
    sv_safe = sv.clamp_min(float(args.svd_eps))
    fisher_logdet = torch.log(sv_safe).sum(dim=1)
    fisher_trace = torch.diagonal(fishers, dim1=1, dim2=2).sum(dim=1)
    fisher_min_sv = sv.min(dim=1).values
    fisher_max_sv = sv.max(dim=1).values

    pup_uncertainty = -fisher_logdet
    pup_uncertainty_rank01 = rank01_high(pup_uncertainty)
    pup_sensitivity_rank01 = rank01_high(fisher_logdet)

    output_dir = args.output_dir or os.path.join(
        dataset.model_path,
        "pup_gaussian_uncertainty",
        os.path.splitext(os.path.basename(checkpoint))[0],
    )
    makedirs(output_dir, exist_ok=True)

    tensors = {
        "fishers": fishers.detach().cpu(),
        "singular_values": sv.detach().cpu(),
        "fisher_logdet": fisher_logdet.detach().cpu(),
        "fisher_trace": fisher_trace.detach().cpu(),
        "fisher_min_sv": fisher_min_sv.detach().cpu(),
        "fisher_max_sv": fisher_max_sv.detach().cpu(),
        "pup_uncertainty": pup_uncertainty.detach().cpu(),
        "pup_uncertainty_rank01": pup_uncertainty_rank01.detach().cpu(),
        "pup_sensitivity_rank01": pup_sensitivity_rank01.detach().cpu(),
        "xyz": gaussians.get_xyz.detach().cpu(),
        "scaling": gaussians.get_scaling.detach().cpu(),
        "opacity": gaussians.get_opacity.detach().cpu(),
    }
    torch.save(tensors, os.path.join(output_dir, "pup_gaussian_uncertainty.pt"))

    csv_path = os.path.join(output_dir, "pup_gaussian_uncertainty.csv")
    arr = torch.stack(
        [
            fisher_logdet,
            pup_uncertainty,
            pup_uncertainty_rank01,
            pup_sensitivity_rank01,
            fisher_trace,
            fisher_min_sv,
            fisher_max_sv,
            gaussians.get_opacity.squeeze(-1),
            torch.prod(gaussians.get_scaling, dim=1),
        ],
        dim=1,
    ).detach().cpu().numpy()
    header = (
        "gaussian_id,fisher_logdet,pup_uncertainty,pup_uncertainty_rank01,"
        "pup_sensitivity_rank01,fisher_trace,fisher_min_sv,fisher_max_sv,opacity,volume"
    )
    ids = np.arange(arr.shape[0], dtype=np.float64)[:, None]
    np.savetxt(csv_path, np.concatenate([ids, arr], axis=1), delimiter=",", header=header, comments="")

    finite_unc = pup_uncertainty[torch.isfinite(pup_uncertainty)]
    finite_logdet = fisher_logdet[torch.isfinite(fisher_logdet)]
    summary = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "checkpoint": checkpoint,
        "first_iter_in_checkpoint": int(first_iter) if isinstance(first_iter, int) else str(first_iter),
        "output_dir": output_dir,
        "num_gaussians": int(n_gaussians),
        "num_train_cameras_used": int(len(cameras)),
        "pool_resolution": int(args.pool_resolution),
        "svd_eps": float(args.svd_eps),
        "fisher_logdet_mean": float(finite_logdet.mean().item()) if finite_logdet.numel() else None,
        "fisher_logdet_min": float(finite_logdet.min().item()) if finite_logdet.numel() else None,
        "fisher_logdet_max": float(finite_logdet.max().item()) if finite_logdet.numel() else None,
        "pup_uncertainty_mean": float(finite_unc.mean().item()) if finite_unc.numel() else None,
        "pup_uncertainty_min": float(finite_unc.min().item()) if finite_unc.numel() else None,
        "pup_uncertainty_max": float(finite_unc.max().item()) if finite_unc.numel() else None,
        "zero_trace_count": int((fisher_trace <= 0).sum().item()),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved tensors: {os.path.join(output_dir, 'pup_gaussian_uncertainty.pt')}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
