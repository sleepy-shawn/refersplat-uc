from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = "LangSplat/data/bed"
        self._model_path = "LangSplat/output/bed" 
        self._language_features_name = "language_features_dim3"
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self._feature_level = -1
        self.data_device = "cuda"
        self.eval = False
        # Cross-scene negatives from `json/<train,test>_json/<frame>.json[negative]`.
        self.use_negative_samples = False
        # Zero-shot eval on perturbation variants. When set to one of
        # {"attribute", "category", "spatial"}, dataset_readers reads
        # from `<scene>/json_perturb_<variant>/...` instead of `json/...`
        # and treats the JSON's `perturbed` array as the negative source.
        # Default empty string = use original `json/` data.
        self.perturb_variant = ""
        # Plan B: inject spatial perturbations as TRAINING negatives. Reads
        # `<scene>/json_perturb_spatial/<train,test>_json/<frame>.json[perturbed]`
        # IN ADDITION to whatever the main path (json/ or json_perturb_<variant>/)
        # provides. Use this WITHOUT perturb_variant for normal training; use
        # WITH perturb_variant="spatial" to stack same-source negatives at eval.
        # If `use_negative_samples` is also True, BOTH cross-scene and spatial
        # negatives are loaded (each independently capped by their max).
        self.use_spatial_negatives = False
        # Held-out preposition control. When non-empty, filters spatial
        # perturbed entries by their `change.from` field. Action determines
        # whether matched entries are excluded (training-time held-out) or
        # included only (eval-time held-out test).
        # Comma-separated phrases, e.g. "in the center of,on the edge of".
        self.spatial_held_out_phrases = ""
        self.spatial_held_out_action = "none"   # {"none","exclude","include_only"}
        # Per-frame caps for negative sample types (avoid one source dominating).
        # -1 = no cap.
        self.max_cross_scene_neg_per_frame = -1
        self.max_spatial_neg_per_frame = -1
        # NEW (2026-05-16): comma-separated list of perturbation variant
        # directories to load as training negatives. The dataset's old
        # base-json `negative` field is deprecated; this is the supported
        # path for training-time negatives going forward.
        # Examples: "attribute,category,spatial,borrow" or "borrow,spatial".
        # Each entry reads `<scene>/json_perturb_<variant>/{train,test}_json/...`
        # and adds the `perturbed[]` items as `is_negative=True`.
        # max_spatial_neg_per_frame still caps per-variant entries.
        self.training_neg_variants = ""
        # Target neg ratio per frame after combining all `training_neg_variants`.
        # -1.0 = no subsampling (legacy: keep every variant's entries; native
        # ramen/figurines/etc. neg ratio is ~41%).
        # When set in [0,1], compute target_neg_per_frame = round(r*pos/(1-r))
        # and stratified-subsample across variants.
        # gRefCOCO reference ratio: 11.6%. Use 0.15 as the default current
        # experiment prior: lower than the previous 0.20 setting, but still
        # above the 0.12 run that collapsed from too little absent signal.
        self.training_neg_target_ratio = 0.15
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        g.lf_path = os.path.join(g.source_path, g.language_features_name)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 60_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        
        self.include_feature = True # Set to False if train the original gs
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.text_language_feature_lr = 0.0025
        self.mlp_lr = 0.0001
        self.cross_attention_lr = 0.0001
        self.language_feature_lr = 0.0025
        # Weight on the contrastive (multi_pos_cross_entropy) com_loss term
        # added to L_mask for positive samples. Was hardcoded to 0.1 in train.py;
        # made into a CLI flag so per-scene tuning is explicit. Default 0.1.
        self.lambda_com = 0.1
        # q_nt no-target query experiments. When enabled, one or more learnable
        # query rows are appended on the Gaussian-query side of PCMI; those rows
        # feed PresentHead and do not participate in rasterization.
        self.use_q_nt = False
        self.q_nt_num_queries = 1
        self.q_nt_pool = "first"          # "first" or "gap"
        # Ablation: remove the learnable pseudo-position f_p_nt from q_nt.
        # Then q_nt is a single learnable semantic query: Q_nt=q_linear(f_nt).
        self.q_nt_no_fp = False
        # Scene-aware fusion for the classification head:
        # concat([q_nt_or_f_u_output, opacity-pooled g, max_score, topk_score_mean])
        # -> PresentHead. Requires use_q_nt or query_concat.
        self.use_scene_aware_fusion = False
        # Cleaner evidence-only variants:
        # - use_topk_evidence_gap: PresentHead sees GAP(g_topk).
        # - use_topk_evidence_fusion: PresentHead sees concat(query_feature, GAP(g_topk)).
        #   g_topk is selected by the same text-response score used for com_loss.
        # - topk_evidence_ratio < 0: reuse the render() ratio, i.e. exactly the
        #   same top-k ratio as the contrastive mean_tensor selection.
        # - fusion_layer_norm: apply non-parametric LayerNorm to each vector before concat.
        self.use_topk_evidence_gap = False
        self.use_topk_evidence_fusion = False
        self.topk_evidence_ratio = -1.0
        self.fusion_layer_norm = False
        self.fusion_query_layer_norm = False
        self.fusion_detach_pooled_g = False
        # BALD-weighted top-k fusion evidence. Requires top-k fusion; render()
        # samples the current and probe-view mask probabilities per Gaussian,
        # computes binary BALD in [0,1], and uses either stable (1-BALD) or
        # uncertain (BALD) weights when pooling g_topk.
        self.use_bald_evidence_weight = False
        self.bald_weight_mode = "stable"      # "stable" or "uncertain"
        self.bald_weight_eps = 0.05
        self.bald_probe_max_angle = 60.0
        self.bald_probe_strategy = "random"   # train default; eval overrides to nearest unless specified
        self.use_uncertain_token = False
        self.uncertain_gamma = 1.0
        self.uncertain_tau = 0.5
        self.lambda_rej = 1.0
        self.lambda_anti_shortcut = 1.0
        self.epsilon_anti_shortcut = 0.1
        # Ablation: which signal feeds the uncertainty-gate query.
        # "fr_plus_fp"     : q_for_u = f_r + f_p              (original, sentence-INVARIANT at inference)
        # "frpost_plus_fp" : q_for_u = f'_r + f_p             (post-PCMI text-conditioned + position)
        # "frpost_only"    : q_for_u = f'_r                   (pure post-PCMI text-conditioned, no position)
        # where f'_r = softmax(QK^T)·V is the cross-attention output BEFORE residual.
        # Main attention path is unchanged across all three modes.
        # Only used when unctoken_arch == "external".
        self.unctoken_query_mode = "fr_plus_fp"
        # Architecture for the uncertain-token gate:
        # "external" (default, our original): a separate W_q_u / W_k_u branch + dot product
        #   with a learnable constant f_u. Gate is computed in parallel to main attention.
        # "inline" (advisor's CLS-style design): f_u is appended to the word sequence W and
        #   participates in the SAME cross-attention softmax. Per-Gaussian uncertainty is
        #   extracted from the attention weight on f_u (× UCT's V contribution) → MLP → u.
        #   More elegant: no extra projection, sentence-aware via joint softmax.
        self.unctoken_arch = "external"
        # --- Multi-View Kendall Aleatoric Loss (MV-Kendall) --------------
        # If enabled, train.py renders K views per iter, derives per-Gaussian
        # uncertainty u_i from the Bernoulli entropy of the cross-view mean
        # mask probability (sampled at each Gaussian's projected 2D location),
        # splats u_i to a 2D image, and applies Kendall & Gal 2017's aleatoric
        # loss form on the main view: BCE / (2u² + ε) + 0.5·log(u² + ε).
        # Replaces the standard mask BCE; Dice / contrastive / L_rej / L_anti
        # are kept unchanged.
        self.use_kendall_aux = False
        self.kendall_K = 2                      # number of views per iter (currently 2 supported)
        self.kendall_eps = 1e-3                 # numerical floor on u² for stability
        self.lambda_kendall = 1.0               # scales the Kendall loss term
        self.kendall_probe_max_angle = 60.0     # max angle (deg) between main and probe camera_center
        self.kendall_probe_detach = True        # if True, probe view forward runs under no_grad
        self.kendall_warmup_iters = 2000        # use plain BCE for first N iters so u_image
                                                # is informative before being trusted
        # --- PresentHead classifier (new architecture) -------------------
        # When --use_present_head is on:
        #   - Renderer computes a present_logit from FFN(weighted-GAP(g)).
        #   - Pooling weight = alpha if uc off; alpha*(1-u).clamp_min(eps) if uc on.
        #   - Train: BCEWithLogits(present_logit, y_present) replaces L_rej/L_anti.
        #   - The legacy soft gate g*(1-u)^γ is DISABLED in this mode.
        #   - Test: pred = zeros if present_logit <= 0 else binary mask.
        self.use_present_head = False
        self.present_head_hidden = 128
        self.present_head_dropout = 0.1
        self.present_head_eps = 0.05       # option-X: weight = α·(1-u).clamp_min(eps)
        self.present_head_threshold = 0.0  # logit threshold; > threshold means present
        self.lambda_classifier = 1.0
        # Weight for no-target samples in PresentHead BCE. y_present=0 gets
        # this weight; present samples keep weight 1.0.
        self.present_negative_weight = 1.0
        # Stage-2 calibration mode: freeze all non-classifier parameters.
        # With --use_gaussian_attr_conv_head this keeps both PresentHead and
        # the attr-conv-former trainable.
        self.present_head_only = False
        # Attribute-conv classifier for many top-k Gaussians. It builds
        # per-Gaussian tokens from [g, xyz, scale, rotation, opacity, SH, UC],
        # downsamples the top-k evidence sequence with Conv1d/pooling, then
        # feeds a small CLS transformer before PresentHead.
        self.use_gaussian_attr_conv_head = False
        self.external_gaussian_uncertainty_path = ""
        self.external_gaussian_uncertainty_key = ""
        self.gaussian_attr_conv_pooled_tokens = 64
        self.gaussian_attr_conv_num_layers = 2
        self.gaussian_attr_conv_num_heads = 4
        self.gaussian_attr_conv_ffn_dim = 256
        self.gaussian_attr_conv_dropout = 0.0
        self.gaussian_attr_conv_kernel_size = 5
        # Score-change refer uncertainty on frozen original ReferSplat.
        self.use_refer_uncertainty = False
        self.refer_uncertainty_only = False
        self.refer_uncertainty_score_only = False
        self.refer_uncertainty_prior_std = 0.0025
        self.refer_uncertainty_kl_weight = 1e-4
        self.refer_uncertainty_score_topk_ratio = 0.03
        self.refer_uncertainty_score_probe_std = 0.0025
        self.refer_uncertainty_score_target_scale = 0.5
        self.refer_uncertainty_score_loss_weight = 1.0
        self.refer_uncertainty_reparam_score_weight = 1.0
        self.refer_uncertainty_log_sigma_min = -5.0
        self.refer_uncertainty_log_sigma_max = 2.0
        self.reset_iter_on_restore = False
        # Variational ReferSplat path adapted from Variational 3DGS:
        # _language_feature remains the trainable deterministic base; delta_mu
        # is the perturbation posterior mean. Render first chooses a top-k gate
        # from the base response, then samples
        #   z = base + gate * (delta_mu + prior_std * exp(log_sigma) * eps)
        # so only responsive Gaussians receive task gradients through offsets.
        self.use_variational_language = False
        self.variational_language_prior_std = 0.0025
        self.variational_language_kl_weight = 1e-4
        self.variational_language_kl_warmup_iters = 2000
        self.variational_language_gate_warmup_iters = 2000
        self.variational_language_offset_lr_scale = 0.1
        # <= 0 means derive from language_feature_lr * offset_lr_scale.
        self.variational_language_mu_lr = 0.0
        self.variational_language_sigma_lr = 0.0
        self.variational_language_log_sigma_min = -5.0
        self.variational_language_log_sigma_max = 2.0
        self.variational_language_eval_samples = 10
        # --- TRUE Kendall heteroscedastic loss (self-learning σ²) ---
        # Replaces the broken `use_kendall_aux` which computed σ² from
        # cross-view entropy under torch.no_grad() (no self-emergence, the
        # 0.5·log σ² regularizer term had zero gradient → placebo).
        #
        # With --use_kendall_self: a SigmaHead (Linear D→1) outputs log σ²
        # per Gaussian from cross_attention features g. We splat to image
        # space and apply the standard Kendall form
        #   L = bce / (2·σ²) + 0.5·log σ²
        # with full gradient through BOTH terms. σ² self-emerges.
        # Mutually exclusive with use_kendall_aux (asserted in train.py).
        self.use_kendall_self = False
        self.lambda_kendall_self = 1.0
        self.kendall_self_warmup_iters = 2000    # plain BCE for first N iters
                                                  # so mask logits are non-random
                                                  # before σ² starts learning
        self.kendall_self_eps = 1e-3
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
   
    cmdlne_string = sys.argv[1:]
    
    cfgfile_string = "Namespace()"
    
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    
    args_cfgfile = eval(cfgfile_string)
    
    merged_dict = vars(args_cfgfile).copy()
    
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    
    return Namespace(**merged_dict)
