

import time
import random
import torch.nn.functional as F
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def min_max_normalize_torch(points):
    min_vals = points.min(dim=0).values  
    max_vals = points.max(dim=0).values  
    
    normalized_points = 2 * (points - min_vals) / (max_vals - min_vals) - 1
    return normalized_points


def _rank_fraction_1d(values):
    if values.numel() <= 1:
        return torch.full_like(values, 0.5, dtype=torch.float32)
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def _spearman_1d(a, b):
    if a is None or b is None or a.numel() <= 1:
        device = a.device if torch.is_tensor(a) else "cuda"
        return torch.tensor(float("nan"), device=device)
    ar = _rank_fraction_1d(a.detach())
    br = _rank_fraction_1d(b.detach())
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = ar.norm() * br.norm()
    if float(denom.item()) <= 1e-12:
        return torch.tensor(float("nan"), device=a.device)
    return (ar * br).sum() / denom


def _normalize_scene_xyz(xyz):
    min_vals = xyz.min(dim=0).values
    max_vals = xyz.max(dim=0).values
    denom = (max_vals - min_vals).clamp_min(1e-6)
    return 2.0 * (xyz - min_vals) / denom - 1.0


def _build_gaussian_attr_tokens(pc, g, indices):
    uc = getattr(pc, "external_gaussian_uncertainty", None)
    if uc is None:
        raise ValueError(
            "--use_gaussian_attr_conv_head requires external Gaussian "
            "uncertainty to be loaded"
        )
    uc = uc.to(device=g.device, dtype=g.dtype).reshape(-1)
    if uc.shape[0] != g.shape[0]:
        raise ValueError(
            f"External uncertainty length mismatch in renderer: got {uc.shape[0]}, "
            f"expected {g.shape[0]}"
        )
    idx = indices.to(device=g.device)
    xyz_norm = _normalize_scene_xyz(pc.get_xyz).to(dtype=g.dtype)
    sh_flat = pc.get_features.reshape(pc.get_features.shape[0], -1).to(dtype=g.dtype)
    attr = torch.cat([
        g[idx],
        xyz_norm[idx],
        pc._scaling[idx].to(dtype=g.dtype),
        pc.get_rotation[idx].to(dtype=g.dtype),
        pc.get_opacity[idx].to(dtype=g.dtype),
        sh_flat[idx],
        uc[idx].unsqueeze(-1),
    ], dim=-1)
    return attr, uc[idx]

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt,
           scaling_modifier = 1.0, override_color = None, sentence=None,
           ratio=0.03, probe_view=None, compute_present_head=True,
           iteration=None, force_variational_language_sample=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
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

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    t_token=pc.get_text(sentence).to("cuda")
    t_token=pc.mlp1(t_token)
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    

    p=pc.mlp3(pc.get_xyz)
    p=F.normalize(p,dim=-1)

    # ----- cross-attention (optionally returning per-Gaussian uncertainty u
    # OR f_u_output for query_concat arch) -----
    use_variational_language = (
        getattr(opt, "use_variational_language", False)
        and getattr(pc, "use_variational_language", False)
    )
    use_unc = getattr(opt, "use_uncertain_token", False) and getattr(pc, "use_uncertain_token", False)
    use_q_nt = getattr(opt, "use_q_nt", False) and getattr(pc, "use_q_nt", False)
    use_scene_aware_fusion = (getattr(opt, "use_scene_aware_fusion", False)
                              and getattr(pc, "use_scene_aware_fusion", False))
    use_topk_evidence_gap = (getattr(opt, "use_topk_evidence_gap", False)
                             and getattr(pc, "use_topk_evidence_gap", False))
    use_topk_evidence_fusion = (getattr(opt, "use_topk_evidence_fusion", False)
                                and getattr(pc, "use_topk_evidence_fusion", False))
    fusion_layer_norm = (getattr(opt, "fusion_layer_norm", False)
                         and getattr(pc, "fusion_layer_norm", False))
    fusion_query_layer_norm = (getattr(opt, "fusion_query_layer_norm", False)
                               and getattr(pc, "fusion_query_layer_norm", False))
    fusion_detach_pooled_g = (getattr(opt, "fusion_detach_pooled_g", False)
                              and getattr(pc, "fusion_detach_pooled_g", False))
    use_bald_evidence_weight = (
        getattr(opt, "use_bald_evidence_weight", False)
        and getattr(pc, "use_bald_evidence_weight", False)
    )
    use_gaussian_attr_conv_head = (
        getattr(opt, "use_gaussian_attr_conv_head", False)
        and getattr(pc, "use_gaussian_attr_conv_head", False)
    )
    use_present_head = getattr(opt, "use_present_head", False) and getattr(pc, "use_present_head", False)
    # When unctoken_arch == "query_concat", cross_attention returns
    # (g, f_u_output) where f_u_output [1, D] is the dedicated PresentHead
    # input (instead of pooled g_global). u_per_gaussian is None.
    is_query_concat = (use_unc
                       and getattr(pc.cross_attention, "unctoken_arch", "external") == "query_concat")

    def _cross_attention_forward(x_input, with_aux=False):
        if with_aux and use_q_nt:
            g_out, qnt_out = pc.cross_attention(x_input, p, t_token, return_q_nt=True)
            return g_out, None, None, qnt_out
        if with_aux and use_unc:
            ca_out = pc.cross_attention(x_input, p, t_token, return_uncertainty=True)
            if is_query_concat:
                g_out, fu_out = ca_out
                return g_out, None, fu_out, None
            g_out, u_out = ca_out
            return g_out, u_out, None, None
        g_out = pc.cross_attention(x_input, p, t_token, return_uncertainty=False)
        return g_out, None, None, None

    variational_language_gate = None
    variational_language_base_topk_score_mean = None
    variational_language_sampled_topk_score_mean = None
    base_topk_indices = None
    if use_variational_language:
        pc.ensure_variational_language_parameters()
        with torch.no_grad():
            x_base = pc.mlp2(pc._language_feature)
            g_base, _, _, _ = _cross_attention_forward(x_base, with_aux=False)
            base_score_values = torch.matmul(g_base, t_token.transpose(-1, -2)).squeeze(0)
            base_score_values = base_score_values.sum(dim=-1)
            base_sorted_indices = torch.argsort(base_score_values, descending=True)
            k_base_topk = max(1, int(base_score_values.shape[0] * ratio))
            base_topk_indices = base_sorted_indices[:k_base_topk]
            variational_language_base_topk_score_mean = base_score_values[base_topk_indices].mean()

        variational_language_gate = torch.zeros(
            (base_score_values.shape[0], 1),
            device=base_score_values.device,
            dtype=pc._language_feature.dtype,
        )
        gate_warmup_iters = int(getattr(opt, "variational_language_gate_warmup_iters", 2000))
        gate_enabled = iteration is None or iteration >= gate_warmup_iters
        if gate_enabled:
            variational_language_gate[base_topk_indices] = 1.0
        language_feature_3d = pc.sample_language_feature(
            sample=torch.is_grad_enabled() or force_variational_language_sample,
            gate=variational_language_gate,
        )
        variational_language_kl = pc.variational_language_kl()
        variational_language_stats = pc.variational_language_stats(
            gate=variational_language_gate,
        )
    else:
        language_feature_3d = pc._language_feature
        variational_language_kl = None
        variational_language_stats = {}

    x=pc.mlp2(language_feature_3d)
    u = None
    f_u_output = None
    qnt_output = None
    g, u, f_u_output, qnt_output = _cross_attention_forward(x, with_aux=True)

    # ----- Scene-level U and (optional) legacy soft gate.
    # U (opacity-weighted mean of u_inline) is ALWAYS computed when u is
    # available, so train.py can apply weak L_rej / L_anti supervision on
    # u_inline even when PresentHead is the primary decision head.
    # The legacy soft gate `g = g * (1-u)^γ` is ONLY applied when
    # use_present_head=False (preserving original inline UCT v2 behavior).
    U = None
    if u is not None:
        alpha = opacity  # [N, 1]
        den = alpha.detach().sum().clamp_min(1e-4)
        U = (alpha * u).sum() / den
        if not use_present_head:
            gamma = getattr(opt, "uncertain_gamma", 1.0)
            # Soft gate applied to the full per-Gaussian feature vector g, so
            # both the rasterized mask logit (features = g·t_token^T) and the
            # contrastive branch (mean_tensor from g[top_k]) are gated.
            g = g * (1.0 - u).clamp(min=0.0, max=1.0).pow(gamma)

    features=torch.matmul(g,t_token.transpose(-1,-2)).squeeze(0)
    features=features.sum(dim=-1,keepdim=True)
    scores = features.squeeze(-1)                                      # [N]
    sorted_indices = torch.argsort(scores, descending=True)
    k_topk = max(1, int(scores.shape[0] * ratio))
    indices = base_topk_indices if base_topk_indices is not None else sorted_indices[:k_topk]
    topk_scores_for_diag = scores[indices]
    max_score = scores.max()
    topk_score_mean = topk_scores_for_diag.mean()
    if use_variational_language:
        variational_language_sampled_topk_score_mean = topk_score_mean
    evidence_ratio = float(getattr(opt, "topk_evidence_ratio", -1.0))
    if evidence_ratio < 0.0:
        evidence_ratio = ratio
    k_evidence = max(1, int(scores.shape[0] * evidence_ratio))
    evidence_indices = sorted_indices[:k_evidence]

    # ----- PresentHead: opacity (and optionally u-)weighted GAP -> FFN -----
    # Output is a scalar logit. logit > 0  -> present, logit <= 0 -> absent.
    # Two pooling modes:
    #   - use_uncertain_token=False : weight_i = alpha_i  (pure baseline GAP)
    #   - use_uncertain_token=True  : weight_i = alpha_i * (1 - u_i).clamp_min(eps)
    #                                  (u-aware pooling; eps = "option X" clamp
    #                                   to prevent weight=0 -> NaN collapse)
    present_logit = None
    bald_per_gaussian = None
    bald_evidence_weight = None
    refer_uncertainty_kl = None
    refer_uncertainty_score_loss = None
    refer_uncertainty_scale_loss = None
    refer_uncertainty_reparam_score_loss = None
    refer_uncertainty_mean = None
    refer_uncertainty_top_mean = None
    refer_uncertainty_rel_mean = None
    refer_uncertainty_rel_top_mean = None
    refer_uncertainty_top_std = None
    refer_uncertainty_mu_mean = None
    refer_uncertainty_mu_top_mean = None
    refer_uncertainty_sample_rel_top_mean = None
    score_sensitivity_top_mean = None
    score_reparam_sensitivity_top_mean = None
    score_target_u_top_mean = None
    score_u_spearman_top = None
    gaussian_attr_conv_topk_count = None
    gaussian_attr_conv_pre_adaptive_tokens = None
    gaussian_attr_conv_pooled_tokens = None
    gaussian_attr_conv_uc_topk_mean = None
    gaussian_attr_conv_uc_topk_std = None

    # ----- TRUE Kendall σ² head (self-learning, vs the deprecated
    # cross-view-entropy use_kendall_aux) -----
    # When --use_kendall_self is on, compute per-Gaussian log σ² from g via
    # SigmaHead and splat to image space. Gradient flows through both terms
    # of the Kendall loss in train.py (no torch.no_grad here).
    log_sigma2_image = None
    if getattr(opt, "use_kendall_self", False) and getattr(pc, "sigma_head", None) is not None:
        log_sigma2_per_gauss = pc.sigma_head(g)                    # [N]
        log_sigma2_image = rasterize_per_gaussian_scalar(
            log_sigma2_per_gauss, viewpoint_camera, pc, pipe, bg_color
        )                                                          # [1, H, W]

    # Contrastive-loss top-K Gaussian selection (mean_tensor → com_loss anchor).
    #
    # NB (2026-05-20): the original implementation called
    #   sorted_indices = torch.argsort(features, descending=True)
    #   indices = sorted_indices[:int(N * ratio)].squeeze(1)
    # where `features` has shape [N, 1] (from .sum(dim=-1, keepdim=True) above).
    # torch.argsort defaults to dim=-1; sorting a singleton dim returns all
    # zeros, so `indices` collapsed to [0, 0, ..., 0] and `mean_tensor` was
    # always g[0] repeated K times. Every ReferSplat result computed before
    # this fix had a broken contrastive loss (com_loss anchored to one
    # arbitrary Gaussian instead of the top-K most-text-similar ones). Bug
    # present since the initial commit by jgq111 (732f6e74, 2025-02-16).
    #
    # Fix: sort over the 1-D score vector.
    selected_tensors = g[indices]
    mean_tensor = torch.mean(selected_tensors, dim=0, keepdim=True)



    rendered_image, language_feature_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        language_feature_precomp = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    if use_bald_evidence_weight and probe_view is not None:
        with torch.no_grad():
            probe_pkg = render(
                probe_view, pc, pipe, bg_color, opt,
                scaling_modifier=scaling_modifier,
                override_color=override_color,
                sentence=sentence,
                ratio=ratio,
                probe_view=None,
                compute_present_head=False,
                iteration=iteration,
                force_variational_language_sample=force_variational_language_sample,
            )
            main_img = language_feature_image.squeeze(0) if language_feature_image.dim() == 3 else language_feature_image
            probe_img = probe_pkg["language_feature_image"]
            probe_img = probe_img.squeeze(0) if probe_img.dim() == 3 else probe_img

            xyz = pc.get_xyz
            proj_main, in_main = project_xyz_to_pixels(xyz, viewpoint_camera)
            proj_probe, in_probe = project_xyz_to_pixels(xyz, probe_view)
            p_main = bilinear_sample_at(torch.sigmoid(main_img), proj_main, in_main)
            p_probe = bilinear_sample_at(torch.sigmoid(probe_img), proj_probe, in_probe)

            valid_main = in_main.to(p_main.dtype)
            valid_probe = in_probe.to(p_probe.dtype)
            denom = (valid_main + valid_probe).clamp_min(1.0)
            p_bar = (p_main * valid_main + p_probe * valid_probe) / denom
            mean_entropy = (
                bernoulli_entropy(p_main) * valid_main
                + bernoulli_entropy(p_probe) * valid_probe
            ) / denom
            bald_per_gaussian = (bernoulli_entropy(p_bar) - mean_entropy).clamp_min(0.0)

            # If a top-k Gaussian is invisible in both views, it is not useful
            # scene evidence for the classifier; treat it as maximally uncertain
            # for the stable direction and maximally weighted for the uncertainty
            # direction.
            both_invisible = (~in_main) & (~in_probe)
            bald_per_gaussian = torch.where(
                both_invisible,
                torch.full_like(bald_per_gaussian, math.log(2.0)),
                bald_per_gaussian,
            )
            bald_norm = (bald_per_gaussian / math.log(2.0)).clamp(0.0, 1.0)
            mode = getattr(opt, "bald_weight_mode", "stable")
            if mode == "stable":
                bald_evidence_weight = 1.0 - bald_norm
            elif mode == "uncertain":
                bald_evidence_weight = bald_norm
            else:
                raise ValueError(
                    f"Unsupported bald_weight_mode={mode!r}; expected 'stable' or 'uncertain'"
                )
            eps_bald = float(getattr(opt, "bald_weight_eps", 0.05))
            bald_evidence_weight = bald_evidence_weight.clamp_min(eps_bald)

    use_refer_uncertainty = (
        getattr(opt, "use_refer_uncertainty", False)
        and getattr(pc, "use_refer_uncertainty", False)
        and getattr(pc, "refer_uncertainty_head", None) is not None
    )
    if use_refer_uncertainty:
        prior_std = float(getattr(opt, "refer_uncertainty_prior_std", 0.0025))
        log_min = float(getattr(opt, "refer_uncertainty_log_sigma_min", -5.0))
        log_max = float(getattr(opt, "refer_uncertainty_log_sigma_max", 2.0))
        language_feature_base = pc._language_feature.detach()
        mu, log_sigma = pc.refer_uncertainty_head(language_feature_base)
        log_sigma = log_sigma.clamp(log_min, log_max)
        sigma_rel = torch.exp(log_sigma)
        posterior_std = prior_std * sigma_rel
        posterior_rms = torch.sqrt(mu.pow(2) + posterior_std.pow(2))
        refer_uncertainty_abs = posterior_rms.norm(dim=-1)
        prior_norm = max(prior_std * math.sqrt(float(posterior_std.shape[-1])), 1e-12)
        refer_uncertainty_rel = refer_uncertainty_abs / prior_norm
        refer_uncertainty_kl = (
            -log_sigma
            + 0.5 * (sigma_rel.pow(2) + (mu / max(prior_std, 1e-12)).pow(2) - 1.0)
        ).sum(dim=-1).mean()

        score_topk_ratio = float(getattr(opt, "refer_uncertainty_score_topk_ratio", 0.03))
        score_topk_ratio = max(score_topk_ratio, 1.0 / max(1, int(scores.shape[0])))
        score_topk = max(1, int(scores.shape[0] * score_topk_ratio))
        score_top_idx = sorted_indices[:score_topk]

        with torch.no_grad():
            probe_std = float(getattr(opt, "refer_uncertainty_score_probe_std", prior_std))
            if probe_std <= 0.0:
                probe_std = prior_std
            language_feature_probe = language_feature_base.clone()
            probe_delta = torch.randn_like(language_feature_probe[score_top_idx]) * probe_std
            language_feature_probe[score_top_idx] = language_feature_probe[score_top_idx] + probe_delta
            x_probe = pc.mlp2(language_feature_probe)
            if use_q_nt:
                g_probe, _ = pc.cross_attention(x_probe, p, t_token, return_q_nt=True)
            elif use_unc:
                ca_probe = pc.cross_attention(x_probe, p, t_token, return_uncertainty=True)
                g_probe = ca_probe[0] if isinstance(ca_probe, tuple) else ca_probe
            else:
                g_probe = pc.cross_attention(x_probe, p, t_token, return_uncertainty=False)
            features_probe = torch.matmul(g_probe, t_token.transpose(-1, -2)).squeeze(0)
            scores_probe = features_probe.sum(dim=-1)
            probe_norm = probe_delta.norm(dim=-1).clamp_min(1e-8)
            sensitivity = (scores_probe[score_top_idx] - scores.detach()[score_top_idx]).abs() / probe_norm
            sensitivity = sensitivity.detach()
            sens_rank = _rank_fraction_1d(sensitivity)
            target_scale = float(getattr(opt, "refer_uncertainty_score_target_scale", 0.5))
            target_u_rel = 1.0 + target_scale * (1.0 - 2.0 * sens_rank)
            target_u_rel = target_u_rel.detach()

        # Reparameterization trick: sample delta = mu + sigma * eps so the
        # score target trains both posterior mean and scale.
        eps = torch.randn_like(posterior_std[score_top_idx])
        reparam_delta = mu[score_top_idx] + posterior_std[score_top_idx] * eps
        sample_update = torch.zeros_like(language_feature_base).index_add(0, score_top_idx, reparam_delta)
        language_feature_reparam = language_feature_base + sample_update
        x_reparam = pc.mlp2(language_feature_reparam)
        if use_q_nt:
            g_reparam, _ = pc.cross_attention(x_reparam, p, t_token, return_q_nt=True)
        elif use_unc:
            ca_reparam = pc.cross_attention(x_reparam, p, t_token, return_uncertainty=True)
            g_reparam = ca_reparam[0] if isinstance(ca_reparam, tuple) else ca_reparam
        else:
            g_reparam = pc.cross_attention(x_reparam, p, t_token, return_uncertainty=False)
        features_reparam = torch.matmul(g_reparam, t_token.transpose(-1, -2)).squeeze(0)
        scores_reparam = features_reparam.sum(dim=-1)
        reparam_norm = reparam_delta.norm(dim=-1).clamp_min(1e-8)
        reparam_u_rel = reparam_norm / prior_norm
        reparam_score_shift = (scores_reparam[score_top_idx] - scores.detach()[score_top_idx]).abs()
        reparam_sensitivity = reparam_score_shift / reparam_norm.detach()

        pred_u_rel = reparam_u_rel
        refer_uncertainty_scale_loss = F.smooth_l1_loss(pred_u_rel, target_u_rel)
        refer_uncertainty_reparam_score_loss = F.smooth_l1_loss(reparam_sensitivity, sensitivity)
        reparam_score_weight = float(getattr(opt, "refer_uncertainty_reparam_score_weight", 1.0))
        refer_uncertainty_score_loss = (
            refer_uncertainty_scale_loss
            + reparam_score_weight * refer_uncertainty_reparam_score_loss
        )
        posterior_top_u_rel = refer_uncertainty_rel[score_top_idx]
        mu_norm = mu.norm(dim=-1)
        refer_uncertainty_mean = refer_uncertainty_abs.mean()
        refer_uncertainty_top_mean = refer_uncertainty_abs[score_top_idx].mean()
        refer_uncertainty_rel_mean = refer_uncertainty_rel.mean()
        refer_uncertainty_rel_top_mean = posterior_top_u_rel.mean()
        refer_uncertainty_top_std = posterior_top_u_rel.std(unbiased=False)
        refer_uncertainty_mu_mean = mu_norm.mean()
        refer_uncertainty_mu_top_mean = mu_norm[score_top_idx].mean()
        refer_uncertainty_sample_rel_top_mean = reparam_u_rel.mean()
        score_sensitivity_top_mean = sensitivity.mean()
        score_reparam_sensitivity_top_mean = reparam_sensitivity.detach().mean()
        score_target_u_top_mean = target_u_rel.mean()
        score_u_spearman_top = _spearman_1d(sensitivity, -posterior_top_u_rel)

    if compute_present_head and use_present_head and pc.present_head is not None:
        qnt_feature = None
        if qnt_output is not None:
            qnt_pool = getattr(opt, "q_nt_pool", getattr(pc, "q_nt_pool", "first"))
            if qnt_pool == "gap":
                qnt_feature = qnt_output.mean(dim=0)                    # [D]
            elif qnt_pool == "first":
                qnt_feature = qnt_output[0]                             # [D]
            else:
                raise ValueError(f"Unsupported q_nt_pool={qnt_pool!r}; expected 'first' or 'gap'")

        if use_gaussian_attr_conv_head:
            if getattr(pc, "gaussian_attr_conv_head", None) is None:
                raise ValueError("--use_gaussian_attr_conv_head requires gaussian_attr_conv_head")
            attr_topk, uc_topk = _build_gaussian_attr_tokens(pc, g, evidence_indices)
            ph_input, attr_stats = pc.gaussian_attr_conv_head(
                attr_topk, return_stats=True
            )
            gaussian_attr_conv_topk_count = attr_stats.get("input_tokens")
            gaussian_attr_conv_pre_adaptive_tokens = attr_stats.get("pre_adaptive_tokens")
            gaussian_attr_conv_pooled_tokens = attr_stats.get("pooled_tokens")
            gaussian_attr_conv_uc_topk_mean = uc_topk.mean()
            gaussian_attr_conv_uc_topk_std = uc_topk.std(unbiased=False)
        elif use_topk_evidence_fusion:
            query_feature = qnt_feature
            if query_feature is None and f_u_output is not None:
                query_feature = f_u_output.squeeze(0)
            if query_feature is None:
                raise ValueError("use_topk_evidence_fusion requires --use_q_nt or query_concat uncertain token")

            topk_g = g[evidence_indices]
            if bald_evidence_weight is not None:
                topk_w = bald_evidence_weight[evidence_indices]
                pooled_g = (topk_w.unsqueeze(-1) * topk_g).sum(dim=0) / topk_w.sum().clamp_min(1e-8)
            else:
                pooled_g = topk_g.mean(dim=0)                           # [D]
            if fusion_detach_pooled_g:
                pooled_g = pooled_g.detach()
            if fusion_layer_norm:
                query_feature = F.layer_norm(query_feature, query_feature.shape)
                pooled_g = F.layer_norm(pooled_g, pooled_g.shape)
            elif fusion_query_layer_norm:
                query_feature = F.layer_norm(query_feature, query_feature.shape)
            ph_input = torch.cat([query_feature, pooled_g], dim=0)        # [2D]
        elif use_scene_aware_fusion:
            query_feature = qnt_feature
            if query_feature is None and f_u_output is not None:
                query_feature = f_u_output.squeeze(0)
            if query_feature is None:
                raise ValueError("use_scene_aware_fusion requires --use_q_nt or query_concat uncertain token")

            alpha_g = opacity.squeeze(-1)                               # [N]
            if u is not None:
                eps_w = float(getattr(opt, "present_head_eps", 0.05))
                u_g = u.squeeze(-1)
                weight = alpha_g * (1.0 - u_g).clamp_min(eps_w)
            else:
                weight = alpha_g
            denom_w = weight.sum().clamp_min(1e-8)
            pooled_g = (weight.unsqueeze(-1) * g).sum(dim=0) / denom_w  # [D]

            score_summary = torch.stack([max_score, topk_score_mean])   # [2]
            ph_input = torch.cat([query_feature, pooled_g, score_summary], dim=0)  # [2D+2]
        elif use_topk_evidence_gap:
            ph_input = g[evidence_indices].mean(dim=0)                   # [D]
            if fusion_layer_norm:
                ph_input = F.layer_norm(ph_input, ph_input.shape)
        elif qnt_feature is not None:
            ph_input = qnt_feature                                      # [D]
        elif f_u_output is not None:
            ph_input = f_u_output.squeeze(0)                            # [D]
        else:
            alpha_g = opacity.squeeze(-1)                               # [N]
            if u is not None:
                eps_w = float(getattr(opt, "present_head_eps", 0.05))
                u_g = u.squeeze(-1)                                     # [N]
                weight = alpha_g * (1.0 - u_g).clamp_min(eps_w)         # [N]
            else:
                weight = alpha_g                                        # [N]
            denom_w = weight.sum().clamp_min(1e-8)
            ph_input = (weight.unsqueeze(-1) * g).sum(dim=0) / denom_w  # [D]
        present_logit = pc.present_head(ph_input)                       # scalar
    present_prob = None if present_logit is None else torch.sigmoid(present_logit)

    return {"render": rendered_image,
            "language_feature_image": language_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "mean_tensor": mean_tensor,
            "U": U,
            "u_per_gaussian": u,
            "present_logit": present_logit,
            "present_prob": present_prob,
            "variational_language_kl": variational_language_kl,
            "variational_language_mu_mean": variational_language_stats.get("mu_mean"),
            "variational_language_mu_abs_mean": variational_language_stats.get("mu_abs_mean"),
            "variational_language_mu_norm_mean": variational_language_stats.get("mu_norm_mean"),
            "variational_language_base_norm_mean": variational_language_stats.get("base_norm_mean"),
            "variational_language_posterior_mean_norm_mean": variational_language_stats.get("posterior_mean_norm_mean"),
            "variational_language_sigma_mean": variational_language_stats.get("sigma_mean"),
            "variational_language_sigma_norm_mean": variational_language_stats.get("sigma_norm_mean"),
            "variational_language_log_sigma_mean": variational_language_stats.get("log_sigma_mean"),
            "variational_language_mean_shift_norm_mean": variational_language_stats.get("mean_shift_norm_mean"),
            "variational_language_active_gate_ratio": variational_language_stats.get("active_gate_ratio"),
            "variational_language_active_mu_norm_mean": variational_language_stats.get("active_mu_norm_mean"),
            "variational_language_inactive_mu_norm_mean": variational_language_stats.get("inactive_mu_norm_mean"),
            "variational_language_active_sigma_mean": variational_language_stats.get("active_sigma_mean"),
            "variational_language_inactive_sigma_mean": variational_language_stats.get("inactive_sigma_mean"),
            "variational_language_base_topk_score_mean": (
                None if variational_language_base_topk_score_mean is None
                else variational_language_base_topk_score_mean.detach()
            ),
            "variational_language_sampled_topk_score_mean": (
                None if variational_language_sampled_topk_score_mean is None
                else variational_language_sampled_topk_score_mean.detach()
            ),
            "max_score": max_score.detach(),
            "topk_score_mean": topk_score_mean.detach(),
            "bald_mean": None if bald_per_gaussian is None else bald_per_gaussian.mean().detach(),
            "bald_topk_mean": None if bald_per_gaussian is None else bald_per_gaussian[evidence_indices].mean().detach(),
            "refer_uncertainty_kl": refer_uncertainty_kl,
            "refer_uncertainty_score_loss": refer_uncertainty_score_loss,
            "refer_uncertainty_scale_loss": refer_uncertainty_scale_loss,
            "refer_uncertainty_reparam_score_loss": refer_uncertainty_reparam_score_loss,
            "refer_uncertainty_mean": None if refer_uncertainty_mean is None else refer_uncertainty_mean.detach(),
            "refer_uncertainty_top_mean": None if refer_uncertainty_top_mean is None else refer_uncertainty_top_mean.detach(),
            "refer_uncertainty_rel_mean": None if refer_uncertainty_rel_mean is None else refer_uncertainty_rel_mean.detach(),
            "refer_uncertainty_rel_top_mean": None if refer_uncertainty_rel_top_mean is None else refer_uncertainty_rel_top_mean.detach(),
            "refer_uncertainty_top_std": None if refer_uncertainty_top_std is None else refer_uncertainty_top_std.detach(),
            "refer_uncertainty_mu_mean": None if refer_uncertainty_mu_mean is None else refer_uncertainty_mu_mean.detach(),
            "refer_uncertainty_mu_top_mean": None if refer_uncertainty_mu_top_mean is None else refer_uncertainty_mu_top_mean.detach(),
            "refer_uncertainty_sample_rel_top_mean": None if refer_uncertainty_sample_rel_top_mean is None else refer_uncertainty_sample_rel_top_mean.detach(),
            "score_sensitivity_top_mean": None if score_sensitivity_top_mean is None else score_sensitivity_top_mean.detach(),
            "score_reparam_sensitivity_top_mean": None if score_reparam_sensitivity_top_mean is None else score_reparam_sensitivity_top_mean.detach(),
            "score_target_u_top_mean": None if score_target_u_top_mean is None else score_target_u_top_mean.detach(),
            "score_u_spearman_top": None if score_u_spearman_top is None else score_u_spearman_top.detach(),
            "gaussian_attr_conv_topk_count": None if gaussian_attr_conv_topk_count is None else gaussian_attr_conv_topk_count.detach(),
            "gaussian_attr_conv_pre_adaptive_tokens": None if gaussian_attr_conv_pre_adaptive_tokens is None else gaussian_attr_conv_pre_adaptive_tokens.detach(),
            "gaussian_attr_conv_pooled_tokens": None if gaussian_attr_conv_pooled_tokens is None else gaussian_attr_conv_pooled_tokens.detach(),
            "gaussian_attr_conv_uc_topk_mean": None if gaussian_attr_conv_uc_topk_mean is None else gaussian_attr_conv_uc_topk_mean.detach(),
            "gaussian_attr_conv_uc_topk_std": None if gaussian_attr_conv_uc_topk_std is None else gaussian_attr_conv_uc_topk_std.detach(),
            "log_sigma2_image": log_sigma2_image}


def render_variational_language_mc(viewpoint_camera, pc: GaussianModel, pipe,
                                   bg_color: torch.Tensor, opt,
                                   scaling_modifier=1.0, override_color=None,
                                   sentence=None, ratio=0.03, probe_view=None,
                                   compute_present_head=True, iteration=None,
                                   samples=None):
    """Monte Carlo inference for variational language features.

    This mirrors Variational 3DGS inference: sample the learned posterior
    multiple times, render multiple masks, then marginalize by averaging
    probabilities. The returned language_feature_image is logit(mean_prob) so
    existing thresholding code can keep using sigmoid(logits).
    """
    use_variational_language = (
        getattr(opt, "use_variational_language", False)
        and getattr(pc, "use_variational_language", False)
    )
    if samples is None:
        samples = int(getattr(opt, "variational_language_eval_samples", 1))
    samples = max(1, int(samples))
    if not use_variational_language or samples <= 1:
        return render(
            viewpoint_camera, pc, pipe, bg_color, opt,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            sentence=sentence,
            ratio=ratio,
            probe_view=probe_view,
            compute_present_head=compute_present_head,
            iteration=iteration,
        )

    probs = []
    logits = []
    first_out = None
    for _ in range(samples):
        out = render(
            viewpoint_camera, pc, pipe, bg_color, opt,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            sentence=sentence,
            ratio=ratio,
            probe_view=probe_view,
            compute_present_head=compute_present_head,
            iteration=iteration,
            force_variational_language_sample=True,
        )
        if first_out is None:
            first_out = out
        sample_logits = out["language_feature_image"]
        logits.append(sample_logits)
        probs.append(torch.sigmoid(sample_logits))

    prob_stack = torch.stack(probs, dim=0)
    logit_stack = torch.stack(logits, dim=0)
    prob_mean = prob_stack.mean(dim=0)
    prob_var = prob_stack.var(dim=0, unbiased=False)
    prob_std = prob_var.sqrt()
    entropy = bernoulli_entropy(prob_mean)
    logit_mean = logit_stack.mean(dim=0)
    logit_var = logit_stack.var(dim=0, unbiased=False)
    mean_logits = torch.logit(prob_mean.clamp(1e-6, 1.0 - 1e-6))

    out = dict(first_out)
    out.update({
        "language_feature_image": mean_logits,
        "variational_language_eval_samples": torch.tensor(
            float(samples), device=prob_mean.device,
        ),
        "variational_language_prob_mean": prob_mean,
        "variational_language_prob_var": prob_var,
        "variational_language_prob_std": prob_std,
        "variational_language_predictive_entropy": entropy,
        "variational_language_logit_mean": logit_mean,
        "variational_language_logit_var": logit_var,
        "variational_language_prob_var_mean": prob_var.mean().detach(),
        "variational_language_prob_std_mean": prob_std.mean().detach(),
        "variational_language_predictive_entropy_mean": entropy.mean().detach(),
        "variational_language_logit_var_mean": logit_var.mean().detach(),
    })
    return out


# ---------------------------------------------------------------------------
# Helpers for Multi-View Kendall Aleatoric Uncertainty (MV-Kendall)
# ---------------------------------------------------------------------------
# These are pure-PyTorch utilities used by train.py when `use_kendall_aux=True`.
# They (a) project per-Gaussian 3D positions to screen-space pixel coords for an
# arbitrary view, (b) bilinearly sample a 2D probability map at those coords,
# and (c) rasterize a per-Gaussian scalar [N] to an image [1, H, W] via the
# standard 3DGS rasterizer with one extra forward call.
#
# References:
#   - GO-PRE (ICML 2026): average marginal predictive entropy, Eq 2
#   - Kendall & Gal (NeurIPS 2017): aleatoric uncertainty loss form
#   - SpheriBED (NeurIPS 2026): per-pixel uncertainty rendering, Eq 10 (spirit)

def project_xyz_to_pixels(xyz, viewpoint_camera):
    """Project [N, 3] world-space points to [N, 2] pixel coords for the given view.

    Returns (pixel_xy, in_view_mask):
        pixel_xy [N, 2] - float pixel coordinates (x, y) in [0, W-1] x [0, H-1]
                          (consistent with grid_sample(align_corners=True))
        in_view_mask [N] bool - True iff the point is inside the camera frustum
                                AND in front of the camera (clip-space w > 0).

    Convention: this codebase uses 3DGS/COLMAP-style camera frame (Y-down,
    Z-forward). `getProjectionMatrix` builds a matrix with `P[1,1] > 0` and
    `P[3,2] = +1`, and `getWorld2View2` builds W2C with `R^T` — both consistent
    with Z-forward. Under this convention, NDC y aligned with image y (top row
    = small ndc_y, bottom row = large ndc_y), so the standard
    `(ndc_y + 1) * 0.5 * (H - 1)` mapping is correct without flipping.
    """
    N = xyz.shape[0]
    ones = torch.ones((N, 1), device=xyz.device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz, ones], dim=-1)              # [N, 4]
    # full_proj_transform is stored as (view @ proj).T in Camera, so left-mul
    # by xyz_hom (row-vector convention) yields clip-space coordinates.
    clip = xyz_hom @ viewpoint_camera.full_proj_transform  # [N, 4]
    w = clip[..., 3:4]                                     # [N, 1]
    # Numerical guard for points exactly at the camera plane.
    safe_w = w.where(w.abs() > 1e-7, torch.full_like(w, 1e-7))
    ndc = clip[..., :3] / safe_w                           # [N, 3], x/y in [-1,1] iff inside frustum

    W_pix = viewpoint_camera.image_width
    H_pix = viewpoint_camera.image_height
    # Use (W-1) / (H-1) so pixel coords are in [0, W-1] x [0, H-1] — matches
    # grid_sample(align_corners=True) downstream.
    pixel_x = (ndc[..., 0] + 1.0) * 0.5 * float(W_pix - 1)
    pixel_y = (ndc[..., 1] + 1.0) * 0.5 * float(H_pix - 1)
    in_view = (
        (ndc[..., 0] >= -1.0)
        & (ndc[..., 0] <= 1.0)
        & (ndc[..., 1] >= -1.0)
        & (ndc[..., 1] <= 1.0)
        & (w.squeeze(-1) > 0.0)
    )
    return torch.stack([pixel_x, pixel_y], dim=-1), in_view


def bilinear_sample_at(image_hw, pixel_xy, in_view_mask, align_corners=True):
    """Bilinearly sample a 2D image at fractional pixel coords.

    image_hw:        [H, W] or [1, H, W] tensor (single-channel)
    pixel_xy:        [N, 2] (x, y) in pixel units
    in_view_mask:    [N] bool, sampled value is forced to 0 for out-of-view points
    Returns:         [N] sampled values
    """
    if image_hw.dim() == 2:
        image_hw = image_hw.unsqueeze(0)            # -> [1, H, W]
    C, H, W = image_hw.shape[-3], image_hw.shape[-2], image_hw.shape[-1]
    # Convert pixel coords to grid_sample normalized coords in [-1, 1]
    denom_x = float(max(W - 1, 1))
    denom_y = float(max(H - 1, 1))
    gx = 2.0 * pixel_xy[..., 0] / denom_x - 1.0
    gy = 2.0 * pixel_xy[..., 1] / denom_y - 1.0
    grid = torch.stack([gx, gy], dim=-1)             # [N, 2]
    grid = grid.unsqueeze(0).unsqueeze(0)            # [1, 1, N, 2]
    img_b = image_hw.unsqueeze(0)                    # [1, C, H, W]
    sampled = F.grid_sample(
        img_b, grid, mode='bilinear', padding_mode='zeros',
        align_corners=align_corners,
    )                                                 # [1, C, 1, N]
    sampled = sampled.squeeze(0).squeeze(1).squeeze(0)  # [N] when C == 1
    if sampled.dim() > 1:
        sampled = sampled.mean(dim=0)                # collapse channels if any
    return sampled * in_view_mask.to(sampled.dtype)


def bernoulli_entropy(p, eps=1e-6):
    """Per-element Bernoulli entropy H(p) = -p log p - (1-p) log(1-p).

    p:  arbitrary-shape tensor of probabilities (expected in [0, 1])
    Returns: same shape, values in [0, log 2 ≈ 0.6931]
    """
    p_c = p.clamp(min=eps, max=1.0 - eps)
    return -(p_c * p_c.log() + (1.0 - p_c) * (1.0 - p_c).log())


def rasterize_per_gaussian_scalar(scalar_per_gauss, viewpoint_camera, pc,
                                   pipe, bg_color):
    """Rasterize a per-Gaussian scalar [N] (or [N, 1]) to a [1, H, W] image
    using the standard 3DGS rasterizer with `language_feature_precomp` set to
    the scalar. Returns the language-feature channel only.

    Used to splat per-Gaussian uncertainty u_i back to image space so it can
    be used as a per-pixel weighting in the Kendall loss.
    """
    if scalar_per_gauss.dim() == 1:
        feature = scalar_per_gauss.unsqueeze(-1)     # [N, 1]
    else:
        feature = scalar_per_gauss

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=True,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = torch.zeros_like(means3D)
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Provide a dummy SH for the (unused) RGB channel.
    shs = pc.get_features

    _, lang_feat_img, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        language_feature_precomp=feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    # lang_feat_img: [C, H, W] where C = feature.shape[-1]
    return lang_feat_img                              # [1, H, W] when feature is [N, 1]


def _camera_forward_world(view):
    """Return the camera's forward direction in world coordinates as a unit
    vector. In 3DGS/COLMAP convention the camera looks along +Z in its own
    frame, so the world-space forward is the third column of C2W (i.e. the
    third row of the stored transposed W2C, or column 2 of inverse-W2C).
    """
    # world_view_transform is stored as W2C^T (Camera.__init__ line 51), so
    # transposing back gives W2C. inverse(W2C) is C2W; its 3rd column is the
    # camera's +Z axis in world.
    w2c = view.world_view_transform.transpose(0, 1)        # back to row-major W2C
    c2w = torch.inverse(w2c)
    fwd = c2w[:3, 2]                                       # +Z in camera → world
    return fwd / (fwd.norm() + 1e-8)


def pick_probe_view(main_view, candidates, max_angle_deg=60.0, rng=None,
                    strategy="random"):
    """Pick a probe view from `candidates` whose forward direction is within
    `max_angle_deg` of `main_view`'s forward direction. Falls back to any
    other view if no candidate is in range.

    Uses the world-space camera forward axis (not the camera_center position),
    so this works for both object-centric and forward-facing capture layouts.
    """
    if rng is None:
        rng = random
    main_fwd = _camera_forward_world(main_view)
    cos_thresh = math.cos(math.radians(max_angle_deg))

    feasible = []
    fallback = []
    for v in candidates:
        if v is main_view:
            continue
        v_fwd = _camera_forward_world(v)
        cos_a = float((main_fwd * v_fwd).sum().item())
        fallback.append((cos_a, v))
        if cos_a >= cos_thresh:
            feasible.append((cos_a, v))
    if not feasible:
        feasible = fallback
    if not feasible:
        return main_view

    if strategy == "nearest":
        return max(feasible, key=lambda item: item[0])[1]
    if strategy == "farthest":
        return min(feasible, key=lambda item: item[0])[1]
    if strategy != "random":
        raise ValueError(
            f"Unsupported probe strategy={strategy!r}; expected 'random', 'nearest', or 'farthest'"
        )
    return rng.choice(feasible)[1]
