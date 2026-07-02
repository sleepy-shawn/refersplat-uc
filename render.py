
import re
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torch.nn as nn
import torch.nn.functional as F
from gaussian_renderer import render_variational_language_mc
import torchvision
import random
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
    


def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args,model=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")
    unc_path = os.path.join(model_path, name, "ours_{}".format(iteration), "unc")
    unc_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "unc_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(unc_path, exist_ok=True)
    makedirs(unc_npy_path, exist_ok=True)
    ans=0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for i in range(len(view.sentence)):
            ans+=1
            sn=view.image_name
            number = re.findall(r'\d+', sn)
            number_int = int(number[0])
            output = render_variational_language_mc(
                view, gaussians, pipeline, background, args,
                sentence=view.sentence[i],
            )
            if not args.include_feature:
                rendering = output["render"]
            else:
                rendering = output["language_feature_image"]
                rendering = torch.sigmoid(rendering)
                rendering = (rendering>=0.5).float()
                # PresentHead gate has priority over the legacy uncertain-token gate.
                if getattr(args, "use_present_head", False) and output.get("present_logit") is not None:
                    threshold = float(getattr(args, "present_head_threshold", 0.0))
                    if output["present_logit"].item() <= threshold:
                        rendering = torch.zeros_like(rendering)
                elif getattr(args, "use_uncertain_token", False) and output.get("U") is not None:
                    tau = getattr(args, "uncertain_tau", 0.5)
                    if output["U"].item() > tau:
                        rendering = torch.zeros_like(rendering)

            is_neg_i = bool(view.is_negative[i]) if i < len(view.is_negative) else False
            if not args.include_feature:
                gt = view.original_image[0:3, :, :]
            elif is_neg_i:
                gt = torch.zeros_like(rendering)
            else:
                gt=view.gt_mask[view.category[i]]
            np.save(os.path.join(render_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),rendering.permute(1,2,0).cpu().numpy())
            np.save(os.path.join(gts_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),gt.permute(1,2,0).cpu().numpy())
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            unc = output.get("variational_language_prob_std")
            if unc is not None:
                np.save(
                    os.path.join(unc_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),
                    unc.permute(1, 2, 0).cpu().numpy(),
                )
                torchvision.utils.save_image(
                    unc,
                    os.path.join(unc_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"),
                )
               
def render_sets(dataset : ModelParams,model_path, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
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
        checkpoint = os.path.join(args.model_path, model_path)
        (model_params, first_iter) = torch.load(checkpoint,map_location=f'cuda:{torch.cuda.current_device()}')
        gaussians.restore(model_params, args, mode='test')
        gaussians.maybe_load_external_gaussian_uncertainty(args)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # if not skip_train:
        #      render_set(dataset.model_path, dataset.source_path, "trainb", iteration, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "testccc", args.iteration, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="chkpnt_cbasetea2514.pth")
    parser.add_argument("--use_uncertain_token", action="store_true")
    parser.add_argument("--uncertain_tau", type=float, default=0.5)
    parser.add_argument("--uncertain_gamma", type=float, default=1.0)
    parser.add_argument("--use_present_head", action="store_true")
    parser.add_argument("--present_head_hidden", type=int, default=128)
    parser.add_argument("--present_head_dropout", type=float, default=0.1)
    parser.add_argument("--present_head_eps", type=float, default=0.05)
    parser.add_argument("--present_head_threshold", type=float, default=0.0)
    parser.add_argument("--use_gaussian_attr_conv_head", action="store_true")
    parser.add_argument("--external_gaussian_uncertainty_path", type=str, default="")
    parser.add_argument("--external_gaussian_uncertainty_key", type=str, default="")
    parser.add_argument("--gaussian_attr_conv_pooled_tokens", type=int, default=64)
    parser.add_argument("--gaussian_attr_conv_num_layers", type=int, default=2)
    parser.add_argument("--gaussian_attr_conv_num_heads", type=int, default=4)
    parser.add_argument("--gaussian_attr_conv_ffn_dim", type=int, default=256)
    parser.add_argument("--gaussian_attr_conv_dropout", type=float, default=0.0)
    parser.add_argument("--gaussian_attr_conv_kernel_size", type=int, default=5)
    parser.add_argument("--use_q_nt", action="store_true")
    parser.add_argument("--q_nt_num_queries", type=int, default=1)
    parser.add_argument("--q_nt_pool", type=str, default="first", choices=["first", "gap"])
    parser.add_argument("--q_nt_no_fp", action="store_true")
    parser.add_argument("--use_scene_aware_fusion", action="store_true")
    parser.add_argument("--use_topk_evidence_gap", action="store_true")
    parser.add_argument("--use_topk_evidence_fusion", action="store_true")
    parser.add_argument("--topk_evidence_ratio", type=float, default=-1.0)
    parser.add_argument("--fusion_layer_norm", action="store_true")
    parser.add_argument("--fusion_query_layer_norm", action="store_true")
    parser.add_argument("--fusion_detach_pooled_g", action="store_true")
    parser.add_argument("--use_bald_evidence_weight", action="store_true")
    parser.add_argument("--bald_weight_mode", type=str, default="stable", choices=["stable", "uncertain"])
    parser.add_argument("--bald_weight_eps", type=float, default=0.05)
    parser.add_argument("--bald_probe_max_angle", type=float, default=60.0)
    parser.add_argument("--bald_probe_strategy", type=str, default="nearest", choices=["random", "nearest", "farthest"])
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
    args = get_combined_args(parser)
    args.include_feature=True
    if args.iteration == -1:
        # Derive the output folder suffix from the checkpoint filename so
        # that `ours_<N>/` always matches the milestone index of the ckpt
        # being rendered (chkpnt_cbasetea251<N>.pth -> N). Falls back to 0
        # if the filename does not match the milestone pattern.
        m = re.search(r"chkpnt_cbasetea251(\d+)\.pth$", args.checkpoint_name)
        args.iteration = int(m.group(1)) if m else 0
    model_path = args.checkpoint_name
    render_sets(model.extract(args), model_path, pipeline.extract(args), args.skip_train, args.skip_test, args)
