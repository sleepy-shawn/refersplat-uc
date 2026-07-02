import os
import math
import random
import torch
import matplotlib.pyplot as plt
from random import randint
from utils.loss_utils import l1_loss, ssim,bce_loss,multi_pos_cross_entropy
from gaussian_renderer import (
    render,
    project_xyz_to_pixels,
    bilinear_sample_at,
    bernoulli_entropy,
    rasterize_per_gaussian_scalar,
    pick_probe_view,
)
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, total_iters):

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree,
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
                              use_gaussian_attr_conv_head=getattr(opt, "use_gaussian_attr_conv_head", False),
                              gaussian_attr_conv_pooled_tokens=getattr(opt, "gaussian_attr_conv_pooled_tokens", 64),
                              gaussian_attr_conv_num_layers=getattr(opt, "gaussian_attr_conv_num_layers", 2),
                              gaussian_attr_conv_num_heads=getattr(opt, "gaussian_attr_conv_num_heads", 4),
                              gaussian_attr_conv_ffn_dim=getattr(opt, "gaussian_attr_conv_ffn_dim", 256),
                              gaussian_attr_conv_dropout=getattr(opt, "gaussian_attr_conv_dropout", 0.0),
                              gaussian_attr_conv_kernel_size=getattr(opt, "gaussian_attr_conv_kernel_size", 5))
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        if getattr(opt, "reset_iter_on_restore", False):
            first_iter = 0
    gaussians.maybe_load_external_gaussian_uncertainty(opt)
    if (getattr(opt, "use_variational_language", False)
            and getattr(opt, "refer_uncertainty_score_only", False)):
        raise ValueError(
            "--use_variational_language is the stage-1 semantic uncertainty path; "
            "do not combine it with --refer_uncertainty_score_only"
        )
    if getattr(opt, "present_head_only", False):
        if gaussians.present_head is None:
            raise ValueError("--present_head_only requires --use_present_head")
        classifier_groups = {"present_head"}
        if getattr(opt, "use_gaussian_attr_conv_head", False):
            if gaussians.gaussian_attr_conv_head is None:
                raise ValueError(
                    "--use_gaussian_attr_conv_head requires the Gaussian attr conv head"
                )
            classifier_groups.add("gaussian_attr_conv_head")
        for group in gaussians.optimizer.param_groups:
            is_classifier_head = group.get("name") in classifier_groups
            if not is_classifier_head:
                group["lr"] = 0.0
            for param in group["params"]:
                param.requires_grad_(is_classifier_head)
    if getattr(opt, "refer_uncertainty_only", False):
        if gaussians.refer_uncertainty_head is None:
            raise ValueError("--refer_uncertainty_only requires --use_refer_uncertainty")
        for group in gaussians.optimizer.param_groups:
            is_refer_uncertainty = group.get("name") == "refer_uncertainty_head"
            if not is_refer_uncertainty:
                group["lr"] = 0.0
            for param in group["params"]:
                param.requires_grad_(is_refer_uncertainty)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    tb_writer = SummaryWriter(log_dir=os.path.join(scene.model_path, "tb")) if TENSORBOARD_FOUND else None

    def log_tb_scalar(name, value, step):
        if tb_writer is None or value is None:
            return
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        tb_writer.add_scalar(name, float(value), step)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Pre-filter training cameras: a camera with no referring expressions
    # provides no training signal and would stall the iter-based outer
    # loop. Fail fast if the whole training set comes back empty so we
    # don't spin forever (e.g. when `gt_mask/` is missing and the loader
    # falls back to the test-only `mask/` directory).
    all_train_cameras = scene.getTrainCameras()
    train_cameras = [c for c in all_train_cameras if len(c.sentence) > 0]
    if len(train_cameras) == 0:
        raise ValueError(
            f"No training cameras with referring expressions were found "
            f"({len(all_train_cameras)} cameras total, none with non-empty "
            f"`sentence`). Check that the dataset's per-frame JSONs reference "
            f"existing mask files and that `gt_mask/` (or the fallback `mask/`) "
            f"contains the expected segmentations."
        )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, total_iters), desc="Training progress")
    # `iteration` counts completed optimization steps. A fresh run starts at 0
    # and a checkpoint stored at step N resumes with iteration=N, so the
    # τ ratio schedule continues smoothly across resumes.
    iteration = first_iter
    ratio = max(0.005, 0.1 * (0.6 ** (iteration // 2000)))
    total_loss = []
    # Save 10 evenly-spaced checkpoints across [0, total_iters], using the
    # original chkpnt_cbasetea251{0..9}.pth naming. Use `>=` so milestones
    # already produced by an earlier run are skipped on resume; otherwise
    # resuming from iteration == milestone[k] would overwrite that same
    # checkpoint after one further step.
    save_milestones = [int(total_iters * (i + 1) / 10) for i in range(10)]
    next_save_idx = 0
    while next_save_idx < len(save_milestones) and iteration >= save_milestones[next_save_idx]:
        next_save_idx += 1

    while iteration < total_iters:
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        text_feature = gaussians.get_text(viewpoint_cam.sentence).to("cuda")
        for i in range(len(viewpoint_cam.sentence)):
            if iteration >= total_iters:
                break
            iter_start.record()
            bald_probe_view = None
            if getattr(opt, "use_bald_evidence_weight", False):
                bald_probe_view = pick_probe_view(
                    viewpoint_cam,
                    train_cameras,
                    max_angle_deg=float(getattr(opt, "bald_probe_max_angle", 60.0)),
                    rng=random,
                    strategy=getattr(opt, "bald_probe_strategy", "random"),
                )
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background, opt,
                sentence=viewpoint_cam.sentence[i], ratio=ratio,
                probe_view=bald_probe_view,
                iteration=iteration,
            )
            language_feature, mean_tensor = render_pkg["language_feature_image"], render_pkg["mean_tensor"]
            if opt.include_feature:
                if getattr(opt, "refer_uncertainty_score_only", False):
                    score_loss = render_pkg.get("refer_uncertainty_score_loss")
                    ru_kl = render_pkg.get("refer_uncertainty_kl")
                    if score_loss is None or ru_kl is None:
                        raise ValueError(
                            "--refer_uncertainty_score_only requires renderer "
                            "score supervision outputs"
                        )
                    loss = (
                        float(getattr(opt, "refer_uncertainty_score_loss_weight", 1.0)) * score_loss
                        + float(getattr(opt, "refer_uncertainty_kl_weight", 1e-4)) * ru_kl
                    )
                    loss.backward()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    iter_end.record()
                    iteration += 1
                    if iteration % 2000 == 0 and ratio > 0.005:
                        ratio = ratio * 0.6
                        if ratio < 0.005:
                            ratio = 0.005
                    with torch.no_grad():
                        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                        if iteration % 10 == 0:
                            log_tb_scalar("train/loss", loss, iteration)
                            log_tb_scalar("refer_uncertainty/score_loss", score_loss, iteration)
                            log_tb_scalar("refer_uncertainty/scale_loss", render_pkg.get("refer_uncertainty_scale_loss"), iteration)
                            log_tb_scalar("refer_uncertainty/reparam_score_loss", render_pkg.get("refer_uncertainty_reparam_score_loss"), iteration)
                            log_tb_scalar("refer_uncertainty/kl", ru_kl, iteration)
                            log_tb_scalar("refer_uncertainty/mean", render_pkg.get("refer_uncertainty_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/top_mean", render_pkg.get("refer_uncertainty_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/rel_mean", render_pkg.get("refer_uncertainty_rel_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/rel_top_mean", render_pkg.get("refer_uncertainty_rel_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/top_std", render_pkg.get("refer_uncertainty_top_std"), iteration)
                            log_tb_scalar("refer_uncertainty/mu_mean", render_pkg.get("refer_uncertainty_mu_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/mu_top_mean", render_pkg.get("refer_uncertainty_mu_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/sample_rel_top_mean", render_pkg.get("refer_uncertainty_sample_rel_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/score_sensitivity_top_mean", render_pkg.get("score_sensitivity_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/score_reparam_sensitivity_top_mean", render_pkg.get("score_reparam_sensitivity_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/score_target_u_top_mean", render_pkg.get("score_target_u_top_mean"), iteration)
                            log_tb_scalar("refer_uncertainty/score_u_spearman_top", render_pkg.get("score_u_spearman_top"), iteration)
                            progress_bar.set_postfix({
                                "Loss": f"{ema_loss_for_log:.{7}f}",
                                "ScoreU": f"{float(score_loss.detach().item()):.{5}f}",
                                "KL": f"{float(ru_kl.detach().item()):.{5}f}",
                                "Ustd": f"{float(render_pkg['refer_uncertainty_top_std'].detach().item()):.{5}f}",
                            })
                            progress_bar.update(10)
                            total_loss.append(ema_loss_for_log)
                    if next_save_idx < len(save_milestones) and iteration >= save_milestones[next_save_idx]:
                        torch.save((gaussians.capture(opt.include_feature), iteration),
                                   scene.model_path + "/chkpnt_cbasetea251" + str(next_save_idx) + ".pth")
                        next_save_idx += 1
                    continue

                is_neg = bool(viewpoint_cam.is_negative[i]) if i < len(viewpoint_cam.is_negative) else False
                y = 1.0 if is_neg else 0.0

                # ----- Bug A fix (2026-05-17): skip mask-shape supervision
                # entirely on NEGATIVE samples. The previous design did
                # `bce_loss(language_feature, zeros_like(...))` which pushed
                # ALL pixel logits toward -∞ on ~60% of training iterations
                # (4 perturbation variants × 15% ≈ 60% negatives), causing a
                # systematic negative bias on every Gaussian's mask vote and
                # crippling T-acc on positives.
                # In the PresentHead framework, the FFN L_classifier handles
                # absent detection — the mask head only needs to learn SHAPE
                # from positive samples.
                # com_loss is positive-only already.
                if is_neg:
                    L_mask = torch.tensor(0.0, device=language_feature.device)
                    gt_mask = None     # for the loss path below; classifier still runs
                else:
                    gt_mask = viewpoint_cam.gt_mask[viewpoint_cam.category[i]].to("cuda")

                # Per-sample mask loss (BCE / Kendall / com) is computed
                # ONLY for positive samples now. Negatives go straight to the
                # PresentHead L_classifier block below.
                # Mutual-exclusion for the two Kendall variants:
                if (getattr(opt, "use_kendall_aux", False)
                        and getattr(opt, "use_kendall_self", False)):
                    raise ValueError(
                        "use_kendall_aux (cross-view fake Kendall) and "
                        "use_kendall_self (TRUE self-learning σ²) are mutually "
                        "exclusive. Pick one."
                    )
                # --- TRUE Kendall (self-learning σ²) -----------------------
                use_kendall_self = (getattr(opt, "use_kendall_self", False)
                                    and not is_neg)
                warmup_self = int(getattr(opt, "kendall_self_warmup_iters", 2000))
                if use_kendall_self and iteration < warmup_self:
                    use_kendall_self = False
                if use_kendall_self and render_pkg.get("log_sigma2_image") is not None:
                    eps_ks = float(getattr(opt, "kendall_self_eps", 1e-3))
                    log_sigma2 = render_pkg["log_sigma2_image"]
                    # Clamp log σ² for numerical stability. Lower bound -2
                    # (codex review #6): σ²_min = e^-2 ≈ 0.135 → BCE multiplier
                    # max ≈ 3.7×. The original -6 lower bound would give 200×
                    # multiplier → gradient explosion early in training.
                    log_sigma2 = log_sigma2.clamp(-2.0, 6.0)
                    sigma2 = torch.exp(log_sigma2)
                    bce_pp = F.binary_cross_entropy_with_logits(
                        language_feature, gt_mask, reduction='none')
                    # Kendall heteroscedastic form. CRITICALLY: no torch.no_grad
                    # anywhere — σ² self-emerges from the two-term balance.
                    # At equilibrium dL/d(log σ²) = 0  →  σ² = bce_pp
                    # (codex review #2 corrected the equilibrium analysis).
                    L_kendall_self = (
                        bce_pp / (2.0 * sigma2 + eps_ks)
                        + 0.5 * log_sigma2
                    ).mean()
                    L_mask = float(getattr(opt, "lambda_kendall_self", 1.0)) * L_kendall_self
                # --- Legacy cross-view fake Kendall (use_kendall_aux) -------
                elif (getattr(opt, "use_kendall_aux", False)
                        and not is_neg
                        and iteration >= int(getattr(opt, "kendall_warmup_iters", 2000))):
                    K = int(getattr(opt, "kendall_K", 2))
                    eps_k = float(getattr(opt, "kendall_eps", 1e-3))
                    max_ang = float(getattr(opt, "kendall_probe_max_angle", 60.0))
                    detach_probe = bool(getattr(opt, "kendall_probe_detach", True))

                    probe_view = pick_probe_view(
                        viewpoint_cam, train_cameras, max_angle_deg=max_ang
                    )
                    # Render probe with same sentence; under no_grad if requested.
                    if detach_probe:
                        with torch.no_grad():
                            probe_pkg = render(
                                probe_view, gaussians, pipe, background, opt,
                                sentence=viewpoint_cam.sentence[i], ratio=ratio,
                                iteration=iteration,
                            )
                            probe_lf = probe_pkg["language_feature_image"]
                    else:
                        probe_pkg = render(
                            probe_view, gaussians, pipe, background, opt,
                            sentence=viewpoint_cam.sentence[i], ratio=ratio,
                            iteration=iteration,
                        )
                        probe_lf = probe_pkg["language_feature_image"]

                    xyz = gaussians.get_xyz
                    # Use no_grad for both projections + bilinear samples so
                    # the multi-view entropy signal acts purely as a fixed
                    # teacher weight (codex M6 / general-purpose M6). The
                    # gradient drives the model through the BCE term only.
                    with torch.no_grad():
                        proj_main, in_main = project_xyz_to_pixels(xyz, viewpoint_cam)
                        proj_probe, in_probe = project_xyz_to_pixels(xyz, probe_view)

                        m_main = torch.sigmoid(
                            language_feature.squeeze(0)
                            if language_feature.dim() == 3 else language_feature)
                        m_probe = torch.sigmoid(
                            probe_lf.squeeze(0)
                            if probe_lf.dim() == 3 else probe_lf)

                        p_main = bilinear_sample_at(m_main, proj_main, in_main)
                        p_probe = bilinear_sample_at(m_probe, proj_probe, in_probe)

                        # Out-of-frustum fallback (M4): a Gaussian invisible in
                        # one or both views shouldn't be treated as "confident
                        # background" (which would make u_i tiny and amplify
                        # the BCE weight catastrophically). Instead, average
                        # only over views where the Gaussian is visible; if
                        # invisible in both, fall back to max-entropy r=0.5.
                        valid_main = in_main.to(p_main.dtype)
                        valid_probe = in_probe.to(p_probe.dtype)
                        denom = (valid_main + valid_probe).clamp_min(1.0)
                        r_bar = (p_main * valid_main + p_probe * valid_probe) / denom
                        both_invisible = (~in_main) & (~in_probe)
                        r_bar = torch.where(
                            both_invisible,
                            torch.full_like(r_bar, 0.5),
                            r_bar,
                        )
                        u_i = bernoulli_entropy(r_bar)        # [N], in [0, log 2]

                        # Splat u to image space using the main view.
                        u_image = rasterize_per_gaussian_scalar(
                            u_i, viewpoint_cam, gaussians, pipe, background,
                        )                                     # [1, H, W]

                    # σ² = (u / log 2)^2 normalized to [0, 1]; clamp away from 0
                    # so the BCE weight 1/(2σ²) is bounded (prevents the early-
                    # iter explosion flagged by both reviewers).
                    sigma = (u_image / math.log(2.0)).clamp(0.0, 1.0)
                    sigma = sigma.clamp_min(0.25)             # min σ → max weight 8x
                    sigma2 = sigma.pow(2)

                    bce_pp = F.binary_cross_entropy_with_logits(
                        language_feature, gt_mask, reduction='none')
                    L_kendall = (
                        bce_pp / (2.0 * sigma2 + eps_k)
                        + 0.5 * (sigma2 + eps_k).log()
                    ).mean()
                    L_mask = opt.lambda_kendall * L_kendall
                elif not is_neg:
                    # Standard BCE only for positives (Bug A fix).
                    L_mask = bce_loss(language_feature, gt_mask)
                # Negatives have L_mask already = 0 from above.

                # com_loss is positive-only (Bug A fix).
                if not is_neg:
                    features = gaussians.mlp1(text_feature)
                    features = torch.mean(features, dim=1)
                    mean_tensor = F.normalize(mean_tensor, dim=1)
                    features = F.normalize(features, dim=1)
                    cosine_similarities = (torch.matmul(mean_tensor, features.T) / 0.1).to("cuda")
                    sentence_tensor = torch.zeros(len(viewpoint_cam.sentence))
                    sentence_tensor[i] = 1
                    current_category = viewpoint_cam.category[i]
                    category_indices = [idx for idx, cat in enumerate(viewpoint_cam.category) if cat == current_category]
                    sentence_tensor[category_indices] = 1
                    sentence_tensor = sentence_tensor.unsqueeze(0).to("cuda")
                    com_loss = multi_pos_cross_entropy(cosine_similarities, sentence_tensor)
                    L_mask = L_mask + getattr(opt, "lambda_com", 0.1) * com_loss

                loss = L_mask
                L_classifier = None
                present_prob_for_log = None
                variational_language_kl = render_pkg.get("variational_language_kl")
                variational_language_kl_weight = None
                if (getattr(opt, "use_variational_language", False)
                        and variational_language_kl is not None):
                    base_kl_weight = float(getattr(opt, "variational_language_kl_weight", 1e-4))
                    warmup_iters = int(getattr(opt, "variational_language_kl_warmup_iters", 2000))
                    if warmup_iters > 0:
                        warmup = min(1.0, float(iteration + 1) / float(warmup_iters))
                    else:
                        warmup = 1.0
                    variational_language_kl_weight = base_kl_weight * warmup
                    loss = loss + variational_language_kl_weight * variational_language_kl

                # ----- PresentHead classifier loss (NEW) -----
                # When --use_present_head is on, the present/absent decision is
                # made by the FFN logit (not by U > tau). Train it with a plain
                # BCE-with-logits against y_present ∈ {0,1} (1 = present).
                # In this mode we explicitly SKIP L_rej and L_anti so u_i is
                # only trained implicitly through its role in the pooling
                # weight (alpha · (1-u)).clamp_min(eps).
                stage1_variational_language = getattr(opt, "use_variational_language", False)
                use_ph = ((not stage1_variational_language)
                          and getattr(opt, "use_present_head", False)
                          and render_pkg.get("present_logit") is not None)
                if use_ph:
                    pl = render_pkg["present_logit"]
                    y_present = torch.tensor(
                        1.0 - y,  # y was 1 for negatives; flip so y_present=1 ⇔ present
                        device=pl.device, dtype=pl.dtype,
                    )
                    neg_w = float(getattr(opt, "present_negative_weight", 1.0))
                    cls_weight = torch.where(
                        y_present > 0.5,
                        torch.ones_like(y_present),
                        torch.full_like(y_present, neg_w),
                    )
                    L_classifier = F.binary_cross_entropy_with_logits(
                        pl, y_present, weight=cls_weight)
                    loss = loss + getattr(opt, "lambda_classifier", 1.0) * L_classifier
                    present_prob_for_log = torch.sigmoid(pl.detach())
                elif ((not stage1_variational_language)
                      and getattr(opt, "use_uncertain_token", False)
                      and render_pkg.get("U") is not None):
                    # Legacy U-based supervision (only when PresentHead is off).
                    U = render_pkg["U"]
                    U_c = U.clamp(min=1e-6, max=1.0 - 1e-6)
                    y_t = torch.tensor(y, device=U.device, dtype=U.dtype)
                    L_rej = -(y_t * torch.log(U_c) + (1.0 - y_t) * torch.log(1.0 - U_c))
                    # Symmetric anti-shortcut:
                    #  - positive (y=0): penalize U > epsilon, push U → 0
                    #  - negative (y=1): penalize U < (1 - epsilon), push U → 1
                    eps = opt.epsilon_anti_shortcut
                    L_anti_pos = (1.0 - y_t) * F.relu(U - eps)
                    L_anti_neg = y_t * F.relu((1.0 - eps) - U)
                    L_anti = L_anti_pos + L_anti_neg
                    loss = loss + opt.lambda_rej * L_rej + opt.lambda_anti_shortcut * L_anti

                loss.backward()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            iter_end.record()
            iteration += 1
            if iteration % 2000 == 0 and ratio > 0.005:
                ratio = ratio * 0.6
                if ratio < 0.005:
                    ratio = 0.005
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    log_tb_scalar("train/loss", loss, iteration)
                    log_tb_scalar("present_head/loss", L_classifier, iteration)
                    log_tb_scalar("present_head/prob", present_prob_for_log, iteration)
                    log_tb_scalar("variational_language/kl", variational_language_kl, iteration)
                    log_tb_scalar("variational_language/kl_weight", variational_language_kl_weight, iteration)
                    log_tb_scalar("variational_language/mu_mean", render_pkg.get("variational_language_mu_mean"), iteration)
                    log_tb_scalar("variational_language/mu_abs_mean", render_pkg.get("variational_language_mu_abs_mean"), iteration)
                    log_tb_scalar("variational_language/mu_norm_mean", render_pkg.get("variational_language_mu_norm_mean"), iteration)
                    log_tb_scalar("variational_language/base_norm_mean", render_pkg.get("variational_language_base_norm_mean"), iteration)
                    log_tb_scalar("variational_language/posterior_mean_norm_mean", render_pkg.get("variational_language_posterior_mean_norm_mean"), iteration)
                    log_tb_scalar("variational_language/sigma_mean", render_pkg.get("variational_language_sigma_mean"), iteration)
                    log_tb_scalar("variational_language/sigma_norm_mean", render_pkg.get("variational_language_sigma_norm_mean"), iteration)
                    log_tb_scalar("variational_language/log_sigma_mean", render_pkg.get("variational_language_log_sigma_mean"), iteration)
                    log_tb_scalar("variational_language/mean_shift_norm_mean", render_pkg.get("variational_language_mean_shift_norm_mean"), iteration)
                    log_tb_scalar("variational_language/active_gate_ratio", render_pkg.get("variational_language_active_gate_ratio"), iteration)
                    log_tb_scalar("variational_language/active_mu_norm_mean", render_pkg.get("variational_language_active_mu_norm_mean"), iteration)
                    log_tb_scalar("variational_language/inactive_mu_norm_mean", render_pkg.get("variational_language_inactive_mu_norm_mean"), iteration)
                    log_tb_scalar("variational_language/active_sigma_mean", render_pkg.get("variational_language_active_sigma_mean"), iteration)
                    log_tb_scalar("variational_language/inactive_sigma_mean", render_pkg.get("variational_language_inactive_sigma_mean"), iteration)
                    log_tb_scalar("variational_language/base_topk_score_mean", render_pkg.get("variational_language_base_topk_score_mean"), iteration)
                    log_tb_scalar("variational_language/sampled_topk_score_mean", render_pkg.get("variational_language_sampled_topk_score_mean"), iteration)
                    log_tb_scalar("gaussian_attr_conv/topk_count", render_pkg.get("gaussian_attr_conv_topk_count"), iteration)
                    log_tb_scalar("gaussian_attr_conv/pre_adaptive_tokens", render_pkg.get("gaussian_attr_conv_pre_adaptive_tokens"), iteration)
                    log_tb_scalar("gaussian_attr_conv/pooled_tokens", render_pkg.get("gaussian_attr_conv_pooled_tokens"), iteration)
                    log_tb_scalar("gaussian_attr_conv/uc_topk_mean", render_pkg.get("gaussian_attr_conv_uc_topk_mean"), iteration)
                    log_tb_scalar("gaussian_attr_conv/uc_topk_std", render_pkg.get("gaussian_attr_conv_uc_topk_std"), iteration)
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                    total_loss.append(ema_loss_for_log)
            if next_save_idx < len(save_milestones) and iteration >= save_milestones[next_save_idx]:
                torch.save((gaussians.capture(opt.include_feature), iteration),
                           scene.model_path + "/chkpnt_cbasetea251" + str(next_save_idx) + ".pth")
                next_save_idx += 1
    progress_bar.close()
    if tb_writer is not None:
        tb_writer.close()
    
if __name__ == "__main__":
    # Set up command line argument parser
    torch.set_default_dtype(torch.float32)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = 'output/teatime/chkpnt30000.pth')
    parser.add_argument("--total_iters", type=int, default=45000,
                        help="Total training iterations (paper §4.2 default 45000). "
                             "Saves 10 evenly-spaced ckpts using chkpnt_cbasetea251{0..9}.pth.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path
    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    # Persist the full args Namespace so render.py / test_metrics.py can
    # rebuild it via get_combined_args(). Required for eval to know which
    # flags (use_uncertain_token, use_negative_samples, ...) were active.
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(args))

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.total_iters)

    print("\nTraining complete.")
