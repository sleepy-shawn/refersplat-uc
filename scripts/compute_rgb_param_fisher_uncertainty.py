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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh


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


def rank01_high(values):
    if values.numel() <= 1:
        return torch.full_like(values, 0.5, dtype=torch.float32)
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def rgb_render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
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
        include_feature=True,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

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

    dummy_feature = torch.zeros(
        (pc.get_xyz.shape[0], 1), dtype=pc.get_xyz.dtype, device=pc.get_xyz.device
    )

    rendered_image, _, radii = rasterizer(
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
    return rendered_image, radii


def zero_selected_grads(pc):
    for param in [
        pc._xyz,
        pc._features_dc,
        pc._features_rest,
        pc._scaling,
        pc._rotation,
        pc._opacity,
    ]:
        if param is not None and param.grad is not None:
            param.grad = None


def grad_energy(param, n_gaussians):
    if param is None or param.grad is None:
        return torch.zeros(n_gaussians, dtype=torch.float32, device="cuda")
    grad = param.grad.detach()
    return grad.float().pow(2).reshape(n_gaussians, -1).sum(dim=1)


def set_requires_grad(pc):
    for param in [
        pc._xyz,
        pc._features_dc,
        pc._features_rest,
        pc._scaling,
        pc._rotation,
        pc._opacity,
    ]:
        if param is not None:
            param.requires_grad_(True)
    if pc._language_feature is not None:
        pc._language_feature.requires_grad_(False)
    for module in [pc.mlp1, pc.mlp2, pc.mlp3, pc.cross_attention]:
        for param in module.parameters():
            param.requires_grad_(False)


def uncertainty_from_energy(energy, eps):
    log_energy = torch.log(energy.clamp_min(float(eps)))
    uncertainty = -log_energy
    return log_energy, uncertainty, rank01_high(uncertainty)


def main():
    parser = ArgumentParser(
        description="Estimate per-Gaussian RGB parameter Fisher diagonal energy for ReferSplat checkpoints."
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--hutchinson_samples", type=int, default=1)
    parser.add_argument("--max_train_cameras", type=int, default=-1)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--fisher_eps", type=float, default=1e-20)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    opt.include_feature = bool(getattr(opt, "include_feature", False))

    checkpoint = args.checkpoint_path or os.path.join(dataset.model_path, args.checkpoint_name)
    if not args.checkpoint_path and not args.checkpoint_name:
        raise ValueError("Pass the explicit best checkpoint with --checkpoint_name or --checkpoint_path.")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    gaussians = build_gaussians(dataset, opt)
    scene = Scene(dataset, gaussians, shuffle=False)
    model_params, first_iter = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, opt, mode="test")
    set_requires_grad(gaussians)

    cameras = scene.getTrainCameras()
    if args.camera_stride > 1:
        cameras = cameras[:: args.camera_stride]
    if args.max_train_cameras > 0:
        cameras = cameras[: args.max_train_cameras]

    n_gaussians = gaussians.get_xyz.shape[0]
    accum = {
        "xyz": torch.zeros(n_gaussians, dtype=torch.float32, device="cuda"),
        "color": torch.zeros(n_gaussians, dtype=torch.float32, device="cuda"),
        "scaling": torch.zeros(n_gaussians, dtype=torch.float32, device="cuda"),
        "rotation": torch.zeros(n_gaussians, dtype=torch.float32, device="cuda"),
        "opacity": torch.zeros(n_gaussians, dtype=torch.float32, device="cuda"),
    }

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    n_probes = 0

    for view in tqdm(cameras, desc=f"RGB param Fisher {os.path.basename(checkpoint)}"):
        for _ in range(int(args.hutchinson_samples)):
            zero_selected_grads(gaussians)
            image, _ = rgb_render(view, gaussians, pipe, background)
            noise = torch.empty_like(image).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            loss = (image * noise).sum()
            loss.backward()
            accum["xyz"] += grad_energy(gaussians._xyz, n_gaussians)
            accum["color"] += (
                grad_energy(gaussians._features_dc, n_gaussians)
                + grad_energy(gaussians._features_rest, n_gaussians)
            )
            accum["scaling"] += grad_energy(gaussians._scaling, n_gaussians)
            accum["rotation"] += grad_energy(gaussians._rotation, n_gaussians)
            accum["opacity"] += grad_energy(gaussians._opacity, n_gaussians)
            n_probes += 1

    zero_selected_grads(gaussians)
    denom = float(max(n_probes, 1))
    for key in accum:
        accum[key] = accum[key] / denom

    full_raw_energy = (
        accum["xyz"] + accum["color"] + accum["scaling"] + accum["rotation"] + accum["opacity"]
    )
    group_sensitivity_ranks = []
    group_log_energy = {}
    for key, energy in accum.items():
        log_energy, _, _ = uncertainty_from_energy(energy, args.fisher_eps)
        group_log_energy[key] = log_energy
        group_sensitivity_ranks.append(rank01_high(log_energy))
    full_rankavg_sensitivity = torch.stack(group_sensitivity_ranks, dim=0).mean(dim=0)
    full_rankavg_uncertainty = 1.0 - full_rankavg_sensitivity

    color_log_energy, color_uncertainty, color_uncertainty_rank01 = uncertainty_from_energy(
        accum["color"], args.fisher_eps
    )
    full_raw_log_energy, full_raw_uncertainty, full_raw_uncertainty_rank01 = uncertainty_from_energy(
        full_raw_energy, args.fisher_eps
    )
    full_rankavg_uncertainty_rank01 = rank01_high(full_rankavg_uncertainty)

    output_dir = args.output_dir or os.path.join(
        dataset.model_path,
        "rgb_param_fisher_uncertainty",
        os.path.splitext(os.path.basename(checkpoint))[0],
    )
    makedirs(output_dir, exist_ok=True)

    tensors = {
        "xyz_fisher_energy": accum["xyz"].detach().cpu(),
        "color_fisher_energy": accum["color"].detach().cpu(),
        "scaling_fisher_energy": accum["scaling"].detach().cpu(),
        "rotation_fisher_energy": accum["rotation"].detach().cpu(),
        "opacity_fisher_energy": accum["opacity"].detach().cpu(),
        "full_raw_fisher_energy": full_raw_energy.detach().cpu(),
        "color_log_fisher_energy": color_log_energy.detach().cpu(),
        "full_raw_log_fisher_energy": full_raw_log_energy.detach().cpu(),
        "full_rankavg_sensitivity": full_rankavg_sensitivity.detach().cpu(),
        "color_uncertainty": color_uncertainty.detach().cpu(),
        "color_uncertainty_rank01": color_uncertainty_rank01.detach().cpu(),
        "full_raw_uncertainty": full_raw_uncertainty.detach().cpu(),
        "full_raw_uncertainty_rank01": full_raw_uncertainty_rank01.detach().cpu(),
        "full_rankavg_uncertainty": full_rankavg_uncertainty.detach().cpu(),
        "full_rankavg_uncertainty_rank01": full_rankavg_uncertainty_rank01.detach().cpu(),
        "xyz": gaussians.get_xyz.detach().cpu(),
        "scaling": gaussians.get_scaling.detach().cpu(),
        "opacity": gaussians.get_opacity.detach().cpu(),
    }
    torch.save(tensors, os.path.join(output_dir, "rgb_param_fisher_uncertainty.pt"))

    csv_path = os.path.join(output_dir, "rgb_param_fisher_uncertainty.csv")
    arr = torch.stack(
        [
            accum["color"],
            color_uncertainty,
            color_uncertainty_rank01,
            full_raw_energy,
            full_raw_uncertainty,
            full_raw_uncertainty_rank01,
            full_rankavg_uncertainty,
            full_rankavg_uncertainty_rank01,
            accum["xyz"],
            accum["scaling"],
            accum["rotation"],
            accum["opacity"],
        ],
        dim=1,
    ).detach().cpu().numpy()
    ids = np.arange(arr.shape[0], dtype=np.float64)[:, None]
    header = (
        "gaussian_id,color_fisher_energy,color_uncertainty,color_uncertainty_rank01,"
        "full_raw_fisher_energy,full_raw_uncertainty,full_raw_uncertainty_rank01,"
        "full_rankavg_uncertainty,full_rankavg_uncertainty_rank01,"
        "xyz_fisher_energy,scaling_fisher_energy,rotation_fisher_energy,opacity_fisher_energy"
    )
    np.savetxt(csv_path, np.concatenate([ids, arr], axis=1), delimiter=",", header=header, comments="")

    def stats(values):
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": float(finite.mean().item()),
            "min": float(finite.min().item()),
            "max": float(finite.max().item()),
        }

    summary = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "checkpoint": checkpoint,
        "first_iter_in_checkpoint": int(first_iter) if isinstance(first_iter, int) else str(first_iter),
        "output_dir": output_dir,
        "method": "Hutchinson diagonal estimate of RGB Jacobian Fisher, E[(J^T v)^2]",
        "hutchinson_samples": int(args.hutchinson_samples),
        "num_train_cameras_used": int(len(cameras)),
        "num_probes": int(n_probes),
        "num_gaussians": int(n_gaussians),
        "fisher_eps": float(args.fisher_eps),
        "color_uncertainty_rank01_stats": stats(color_uncertainty_rank01),
        "full_raw_uncertainty_rank01_stats": stats(full_raw_uncertainty_rank01),
        "full_rankavg_uncertainty_rank01_stats": stats(full_rankavg_uncertainty_rank01),
        "zero_color_energy_count": int((accum["color"] <= 0).sum().item()),
        "zero_full_raw_energy_count": int((full_raw_energy <= 0).sum().item()),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved tensors: {os.path.join(output_dir, 'rgb_param_fisher_uncertainty.pt')}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
