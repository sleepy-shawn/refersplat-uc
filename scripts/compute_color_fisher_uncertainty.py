import json
import math
import os
import random
import sys
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh


def rank01_high(values):
    if values.numel() <= 1:
        return torch.full_like(values, 0.5, dtype=torch.float32)
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


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
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return rendered_image, radii


def zero_feature_grads(gaussians):
    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None


def accumulate_color_fisher(view, gaussians, pipeline, background, accum, generator, probes):
    for _ in range(probes):
        zero_feature_grads(gaussians)
        image, _ = rgb_render(view, gaussians, pipeline, background)
        noise = torch.empty(
            image.shape, dtype=image.dtype, device=image.device
        ).bernoulli_(0.5, generator=generator)
        noise.mul_(2.0).sub_(1.0)
        image.backward(gradient=noise)

        dc_grad = gaussians._features_dc.grad
        rest_grad = gaussians._features_rest.grad
        if dc_grad is not None:
            accum += dc_grad.detach().pow(2).sum(dim=(1, 2))
        if rest_grad is not None:
            accum += rest_grad.detach().pow(2).sum(dim=(1, 2))
        zero_feature_grads(gaussians)


def freeze_non_color_params(gaussians):
    for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_language_feature"):
        value = getattr(gaussians, name, None)
        if value is not None and torch.is_tensor(value):
            value.requires_grad_(False)
    gaussians._features_dc.requires_grad_(True)
    gaussians._features_rest.requires_grad_(True)


def main():
    parser = ArgumentParser(description="Compute color-coefficient Fisher diagonal uncertainty for ReferSplat checkpoints.")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--geometry_pup_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--hutchinson_probes", type=int, default=4)
    parser.add_argument("--hutchinson_seed", type=int, default=20260627)
    parser.add_argument("--max_train_cameras", type=int, default=-1)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    if not args.checkpoint_name and not args.checkpoint_path:
        raise ValueError("Use the project protocol: pass the explicit best checkpoint via --checkpoint_name or --checkpoint_path.")

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    gaussians = build_gaussians(dataset, args)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint = args.checkpoint_path or os.path.join(dataset.model_path, args.checkpoint_name)
    model_params, first_iter = torch.load(checkpoint, map_location=f"cuda:{torch.cuda.current_device()}", weights_only=False)
    gaussians.restore(model_params, args, mode="test")
    freeze_non_color_params(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    if args.camera_stride > 1:
        cameras = cameras[:: args.camera_stride]
    if args.max_train_cameras > 0:
        cameras = cameras[: args.max_train_cameras]

    n_gaussians = gaussians.get_xyz.shape[0]
    color_fisher_diag_sum = torch.zeros(n_gaussians, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(args.hutchinson_seed))

    with torch.enable_grad():
        for view in tqdm(cameras, desc=f"Color Fisher {os.path.basename(checkpoint)}"):
            accumulate_color_fisher(
                view,
                gaussians,
                pipe,
                background,
                color_fisher_diag_sum,
                generator,
                int(args.hutchinson_probes),
            )

    color_fisher_diag_sum = color_fisher_diag_sum / max(1, int(args.hutchinson_probes))
    color_fisher_log = torch.log(color_fisher_diag_sum.clamp_min(float(args.eps)))
    color_uncertainty = -color_fisher_log
    color_sensitivity_rank01 = rank01_high(color_fisher_log)
    color_uncertainty_rank01 = rank01_high(color_uncertainty)

    tensors = {
        "color_fisher_diag_sum": color_fisher_diag_sum.detach().cpu(),
        "color_fisher_log": color_fisher_log.detach().cpu(),
        "color_uncertainty": color_uncertainty.detach().cpu(),
        "color_uncertainty_rank01": color_uncertainty_rank01.detach().cpu(),
        "color_sensitivity_rank01": color_sensitivity_rank01.detach().cpu(),
        "xyz": gaussians.get_xyz.detach().cpu(),
        "opacity": gaussians.get_opacity.detach().cpu(),
    }

    combined_uncertainty_rank01 = None
    if args.geometry_pup_path:
        geometry = torch.load(args.geometry_pup_path, map_location="cpu", weights_only=False)
        if "pup_sensitivity_rank01" in geometry:
            geom_sensitivity = geometry["pup_sensitivity_rank01"].float()
        elif "fisher_logdet" in geometry:
            geom_sensitivity = rank01_high(geometry["fisher_logdet"].float())
        else:
            raise KeyError("geometry_pup_path must contain pup_sensitivity_rank01 or fisher_logdet")
        if geom_sensitivity.numel() != n_gaussians:
            raise ValueError(
                f"Geometry score count {geom_sensitivity.numel()} does not match Gaussian count {n_gaussians}"
            )
        combined_sensitivity_rank01 = 0.5 * (
            geom_sensitivity + color_sensitivity_rank01.detach().cpu()
        )
        combined_uncertainty = 1.0 - combined_sensitivity_rank01
        combined_uncertainty_rank01 = rank01_high(combined_uncertainty)
        tensors["combined_sensitivity_rank01"] = combined_sensitivity_rank01
        tensors["combined_uncertainty_rank01"] = combined_uncertainty_rank01

    output_dir = args.output_dir or os.path.join(
        dataset.model_path,
        "pup_gaussian_uncertainty",
        f"color_fisher_{os.path.splitext(os.path.basename(checkpoint))[0]}",
    )
    makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "color_fisher_uncertainty.pt")
    torch.save(tensors, output_path)

    finite_log = color_fisher_log[torch.isfinite(color_fisher_log)]
    summary = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "checkpoint": checkpoint,
        "first_iter_in_checkpoint": int(first_iter) if isinstance(first_iter, int) else str(first_iter),
        "geometry_pup_path": args.geometry_pup_path,
        "output_dir": output_dir,
        "output_path": output_path,
        "num_gaussians": int(n_gaussians),
        "num_train_cameras_used": int(len(cameras)),
        "hutchinson_probes": int(args.hutchinson_probes),
        "hutchinson_seed": int(args.hutchinson_seed),
        "color_fisher_log_mean": float(finite_log.mean().item()) if finite_log.numel() else None,
        "color_fisher_log_min": float(finite_log.min().item()) if finite_log.numel() else None,
        "color_fisher_log_max": float(finite_log.max().item()) if finite_log.numel() else None,
        "zero_color_fisher_count": int((color_fisher_diag_sum <= 0).sum().item()),
        "has_combined_uncertainty": combined_uncertainty_rank01 is not None,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved tensors: {output_path}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
