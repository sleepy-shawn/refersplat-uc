
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import math
from .cross_attention import (
    MLP1, MLP2, MLP3, CrossAttention, SigmaHead, ReferUncertaintyHead,
)
from uncertainty.fisher import load_uncertainty_tensor
from uncertainty.gaussian_tokens import gaussian_attr_dim
from uncertainty.present_classifier import GaussianAttrConvPoolFormerHead, PresentHead

                       
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, use_uncertain_token : bool = False,
                 unctoken_query_mode : str = "fr_plus_fp",
                 unctoken_arch : str = "external",
                 use_present_head : bool = False,
                 present_head_hidden : int = 128,
                 present_head_dropout : float = 0.1,
                 use_kendall_self : bool = False,
                 use_q_nt : bool = False,
                 q_nt_num_queries : int = 1,
                 q_nt_pool : str = "first",
                 q_nt_no_fp : bool = False,
                 use_scene_aware_fusion : bool = False,
                 use_topk_evidence_gap : bool = False,
                 use_topk_evidence_fusion : bool = False,
                 fusion_layer_norm : bool = False,
                 fusion_query_layer_norm : bool = False,
                 fusion_detach_pooled_g : bool = False,
                 use_bald_evidence_weight : bool = False,
                 use_refer_uncertainty : bool = False,
                 use_variational_language : bool = False,
                 variational_language_prior_std : float = 0.0025,
                 variational_language_log_sigma_min : float = -5.0,
                 variational_language_log_sigma_max : float = 2.0,
                 use_gaussian_attr_conv_head : bool = False,
                 gaussian_attr_conv_pooled_tokens : int = 64,
                 gaussian_attr_conv_num_layers : int = 2,
                 gaussian_attr_conv_num_heads : int = 4,
                 gaussian_attr_conv_ffn_dim : int = 256,
                 gaussian_attr_conv_dropout : float = 0.0,
                 gaussian_attr_conv_kernel_size : int = 5):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.use_uncertain_token = use_uncertain_token
        self.unctoken_query_mode = unctoken_query_mode
        self.unctoken_arch = unctoken_arch
        self.use_present_head = use_present_head
        self.present_head_hidden = present_head_hidden
        self.present_head_dropout = present_head_dropout
        self.use_kendall_self = use_kendall_self
        self.use_q_nt = use_q_nt
        self.q_nt_num_queries = int(q_nt_num_queries)
        self.q_nt_pool = q_nt_pool
        self.q_nt_no_fp = bool(q_nt_no_fp)
        self.use_scene_aware_fusion = use_scene_aware_fusion
        self.use_topk_evidence_gap = use_topk_evidence_gap
        self.use_topk_evidence_fusion = use_topk_evidence_fusion
        self.fusion_layer_norm = fusion_layer_norm
        self.fusion_query_layer_norm = fusion_query_layer_norm
        self.fusion_detach_pooled_g = fusion_detach_pooled_g
        self.use_bald_evidence_weight = use_bald_evidence_weight
        self.use_refer_uncertainty = bool(use_refer_uncertainty)
        self.use_variational_language = bool(use_variational_language)
        self.variational_language_prior_std = float(variational_language_prior_std)
        self.variational_language_log_sigma_min = float(variational_language_log_sigma_min)
        self.variational_language_log_sigma_max = float(variational_language_log_sigma_max)
        self.use_gaussian_attr_conv_head = bool(use_gaussian_attr_conv_head)
        self.gaussian_attr_conv_pooled_tokens = int(gaussian_attr_conv_pooled_tokens)
        self.gaussian_attr_conv_num_layers = int(gaussian_attr_conv_num_layers)
        self.gaussian_attr_conv_num_heads = int(gaussian_attr_conv_num_heads)
        self.gaussian_attr_conv_ffn_dim = int(gaussian_attr_conv_ffn_dim)
        self.gaussian_attr_conv_dropout = float(gaussian_attr_conv_dropout)
        self.gaussian_attr_conv_kernel_size = int(gaussian_attr_conv_kernel_size)
        self.external_gaussian_uncertainty = None
        self.external_gaussian_uncertainty_path = ""
        self.external_gaussian_uncertainty_key = ""
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._language_feature = None
        self._language_feature_delta_mu = None
        self._language_feature_log_sigma = None
        self.feature_project=None 
        self.text_language_feature =torch.empty(0)
        self.mlp2=MLP2(16,128).to("cuda")
        self.mlp3=MLP3(3,128).to("cuda")
        self.mlp1=MLP1(1024,128).to("cuda")

        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.scheduler = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.setup_functions()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained("bert-large-uncased").to("cuda")
        self.model=model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.cross_attention=CrossAttention(dim=128, num_heads=1,
                                            use_uncertain_token=use_uncertain_token,
                                            unctoken_query_mode=unctoken_query_mode,
                                            unctoken_arch=unctoken_arch,
                                            use_q_nt=use_q_nt,
                                            q_nt_num_queries=self.q_nt_num_queries,
                                            q_nt_no_fp=self.q_nt_no_fp).to("cuda")

        # Present/absent classifier head. Only instantiated when requested;
        # otherwise downstream code skips it and uses the legacy U-threshold.
        self.present_head = None
        if self.use_present_head:
            if self.use_gaussian_attr_conv_head:
                present_head_dim = 128
            elif self.use_scene_aware_fusion:
                present_head_dim = 258
            elif self.use_topk_evidence_fusion:
                present_head_dim = 256
            else:
                present_head_dim = 128
            self.present_head = PresentHead(
                D=present_head_dim, hidden=present_head_hidden, p_drop=present_head_dropout,
            ).to("cuda")

        self.gaussian_attr_conv_head = None
        if self.use_gaussian_attr_conv_head:
            attr_dim = gaussian_attr_dim(self.max_sh_degree)
            self.gaussian_attr_conv_head = GaussianAttrConvPoolFormerHead(
                attr_dim=attr_dim,
                D=128,
                pooled_tokens=self.gaussian_attr_conv_pooled_tokens,
                num_self_layers=self.gaussian_attr_conv_num_layers,
                num_self_heads=self.gaussian_attr_conv_num_heads,
                ffn_dim=self.gaussian_attr_conv_ffn_dim,
                p_drop=self.gaussian_attr_conv_dropout,
                kernel_size=self.gaussian_attr_conv_kernel_size,
            ).to("cuda")

        # TRUE Kendall σ² head — only instantiated when requested. Outputs
        # log σ² per Gaussian which gets splat to image space at render time.
        self.sigma_head = None
        if self.use_kendall_self:
            self.sigma_head = SigmaHead(D=128).to("cuda")
        self.refer_uncertainty_head = None
        if self.use_refer_uncertainty:
            self.refer_uncertainty_head = ReferUncertaintyHead(D=16).to("cuda")
    def get_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        with torch.no_grad():
          outputs = self.model(**inputs)
          outputs=outputs[0][:,1:-1,:]
        return outputs
    
    def capture(self, include_feature=False):
        if include_feature:
            base_tuple = (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self.mlp1.state_dict(),
                self.mlp2.state_dict(),
                self.mlp3.state_dict(),
                self.cross_attention.state_dict(),
            )
            head_bundle = {
                "__refsplat_head_bundle__": 1,
                "present_head": None if self.present_head is None else self.present_head.state_dict(),
                "sigma_head": None if self.sigma_head is None else self.sigma_head.state_dict(),
                "refer_uncertainty_head": (
                    None if self.refer_uncertainty_head is None
                    else self.refer_uncertainty_head.state_dict()
                ),
                "gaussian_attr_conv_head": (
                    None if self.gaussian_attr_conv_head is None
                    else self.gaussian_attr_conv_head.state_dict()
                ),
                "variational_language_delta_mu": (
                    None if self._language_feature_delta_mu is None
                    else self._language_feature_delta_mu.detach()
                ),
                "variational_language_log_sigma": (
                    None if self._language_feature_log_sigma is None
                    else self._language_feature_log_sigma.detach()
                ),
            }
            if any(v is not None for k, v in head_bundle.items() if k != "__refsplat_head_bundle__"):
                return base_tuple + (head_bundle,)
            return base_tuple
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )            
    
    def restore(self, model_args, training_args, mode='train'):
        present_head_params = None
        sigma_head_params = None
        refer_uncertainty_params = None
        gaussian_attr_conv_params = None
        variational_language_delta_mu = None
        variational_language_log_sigma = None
        if (len(model_args) == 18
                and isinstance(model_args[-1], dict)
                and model_args[-1].get("__refsplat_head_bundle__") == 1):
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._language_feature,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            mlp1_params,
            mlp2_params,
            mlp3_params,
            cross_attention_params,
            head_bundle,
            ) = model_args
            self.mlp1.load_state_dict(mlp1_params)
            self.mlp2.load_state_dict(mlp2_params)
            self.mlp3.load_state_dict(mlp3_params)
            self.cross_attention.load_state_dict(cross_attention_params, strict=False)
            present_head_params = head_bundle.get("present_head")
            sigma_head_params = head_bundle.get("sigma_head")
            refer_uncertainty_params = head_bundle.get("refer_uncertainty_head")
            gaussian_attr_conv_params = head_bundle.get("gaussian_attr_conv_head")
            variational_language_delta_mu = (
                head_bundle.get("variational_language_delta_mu")
                if "variational_language_delta_mu" in head_bundle
                else head_bundle.get("variational_language_offset_mu")
            )
            variational_language_log_sigma = head_bundle.get("variational_language_log_sigma")
            if self.present_head is not None and present_head_params is not None:
                self.present_head.load_state_dict(present_head_params, strict=False)
            if self.sigma_head is not None and sigma_head_params is not None:
                self.sigma_head.load_state_dict(sigma_head_params, strict=False)
            if self.refer_uncertainty_head is not None and refer_uncertainty_params is not None:
                self.refer_uncertainty_head.load_state_dict(refer_uncertainty_params, strict=False)
            if self.gaussian_attr_conv_head is not None and gaussian_attr_conv_params is not None:
                self.gaussian_attr_conv_head.load_state_dict(gaussian_attr_conv_params, strict=False)
        elif len(model_args) == 19:
            # Newest existing format: 17-base + PresentHead + SigmaHead.
            # In this branch, 17-base + None + SigmaHead + ReferU is also len 20,
            # so len 19 remains unambiguous for the pre-existing format.
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._language_feature,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            mlp1_params,
            mlp2_params,
            mlp3_params,
            cross_attention_params,
            present_head_params,
            sigma_head_params,
            ) = model_args
            self.mlp1.load_state_dict(mlp1_params)
            self.mlp2.load_state_dict(mlp2_params)
            self.mlp3.load_state_dict(mlp3_params)
            self.cross_attention.load_state_dict(cross_attention_params, strict=False)
            if self.present_head is not None and present_head_params is not None:
                self.present_head.load_state_dict(present_head_params, strict=False)
            if self.sigma_head is not None and sigma_head_params is not None:
                self.sigma_head.load_state_dict(sigma_head_params, strict=False)
        elif len(model_args) == 20:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._language_feature,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            mlp1_params,
            mlp2_params,
            mlp3_params,
            cross_attention_params,
            present_head_params,
            sigma_head_params,
            refer_uncertainty_params,
            ) = model_args
            self.mlp1.load_state_dict(mlp1_params)
            self.mlp2.load_state_dict(mlp2_params)
            self.mlp3.load_state_dict(mlp3_params)
            self.cross_attention.load_state_dict(cross_attention_params, strict=False)
            if self.present_head is not None and present_head_params is not None:
                self.present_head.load_state_dict(present_head_params, strict=False)
            if self.sigma_head is not None and sigma_head_params is not None:
                self.sigma_head.load_state_dict(sigma_head_params, strict=False)
            if self.refer_uncertainty_head is not None and refer_uncertainty_params is not None:
                self.refer_uncertainty_head.load_state_dict(refer_uncertainty_params, strict=False)
        elif len(model_args) == 18:
            # Existing format: includes PresentHead at index 17. For the
            # score-only ReferU experiment on original len-17 checkpoints, the
            # appended slot is the ReferU head when no PresentHead is enabled.
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._language_feature,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            mlp1_params,
            mlp2_params,
            mlp3_params,
            cross_attention_params,
            present_head_params,
            ) = model_args
            self.mlp1.load_state_dict(mlp1_params)
            self.mlp2.load_state_dict(mlp2_params)
            self.mlp3.load_state_dict(mlp3_params)
            self.cross_attention.load_state_dict(cross_attention_params, strict=False)
            if (self.refer_uncertainty_head is not None
                    and self.present_head is None
                    and self.sigma_head is None
                    and present_head_params is not None):
                self.refer_uncertainty_head.load_state_dict(present_head_params, strict=False)
            elif self.present_head is not None and present_head_params is not None:
                self.present_head.load_state_dict(present_head_params, strict=False)
        elif len(model_args) == 17:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._language_feature,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            #self.text_language_feature,
            mlp1_params,
            mlp2_params,
            mlp3_params,
            cross_attention_params,
            ) = model_args
            self.mlp1.load_state_dict(mlp1_params)
            self.mlp2.load_state_dict(mlp2_params)
            self.mlp3.load_state_dict(mlp3_params)
            # strict=False so a baseline checkpoint (no uncertain_token params)
            # can be loaded into a uncertain-token-enabled CrossAttention.
            self.cross_attention.load_state_dict(cross_attention_params, strict=False)
        elif len(model_args) == 11: 
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._language_feature,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif len(model_args) == 12: 
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
           
        if variational_language_delta_mu is not None:
            self._language_feature_delta_mu = nn.Parameter(
                variational_language_delta_mu.detach().to("cuda").requires_grad_(True)
            )
        if variational_language_log_sigma is not None:
            self._language_feature_log_sigma = nn.Parameter(
                variational_language_log_sigma.detach().to("cuda").requires_grad_(True)
            )
        if self.use_variational_language and self._language_feature is not None:
            self.ensure_variational_language_parameters()
        
        if mode == 'train':
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
        

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature(self):
        if self._language_feature is not None:
            return self._language_feature
        else:
            raise ValueError('没有设置language feature')

    def load_external_gaussian_uncertainty(self, path, key):
        if not path:
            raise ValueError(
                "--use_gaussian_attr_conv_head requires "
                "--external_gaussian_uncertainty_path"
            )
        values, key = load_uncertainty_tensor(path, key)
        if self._xyz.numel() > 0 and values.shape[0] != self._xyz.shape[0]:
            raise ValueError(
                f"External uncertainty length mismatch: got {values.shape[0]}, "
                f"expected {self._xyz.shape[0]} Gaussians"
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"External uncertainty contains NaN/Inf values: {path}")
        self.external_gaussian_uncertainty = values.to("cuda")
        self.external_gaussian_uncertainty_path = path
        self.external_gaussian_uncertainty_key = key
        print(
            f"[GaussianAttrConv] loaded external uncertainty key={key} "
            f"shape={tuple(values.shape)} from {path}"
        )

    def maybe_load_external_gaussian_uncertainty(self, args):
        if not self.use_gaussian_attr_conv_head:
            return
        path = getattr(args, "external_gaussian_uncertainty_path", "")
        key = getattr(args, "external_gaussian_uncertainty_key", "")
        self.load_external_gaussian_uncertainty(path, key)

    def ensure_variational_language_parameters(self):
        if self._language_feature is None:
            raise ValueError("Variational language requires _language_feature to be initialized")
        if (self._language_feature_delta_mu is None
                or self._language_feature_delta_mu.shape != self._language_feature.shape):
            delta_mu = torch.zeros_like(self._language_feature, device=self._language_feature.device)
            self._language_feature_delta_mu = nn.Parameter(delta_mu.requires_grad_(True))
        if (self._language_feature_log_sigma is None
                or self._language_feature_log_sigma.shape != self._language_feature.shape):
            log_sigma = torch.zeros_like(self._language_feature, device=self._language_feature.device)
            self._language_feature_log_sigma = nn.Parameter(log_sigma.requires_grad_(True))

    def _format_variational_gate(self, gate):
        if gate is None:
            return None
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        return gate.to(device=self._language_feature.device, dtype=self._language_feature.dtype)

    def sample_language_feature(self, sample=True, gate=None):
        if not self.use_variational_language:
            return self.get_language_feature
        self.ensure_variational_language_parameters()
        base = self._language_feature
        mu = self._language_feature_delta_mu
        log_sigma = self._language_feature_log_sigma.clamp(
            self.variational_language_log_sigma_min,
            self.variational_language_log_sigma_max,
        )
        std = self.variational_language_prior_std * torch.exp(log_sigma)
        if sample:
            eps = torch.randn_like(std)
            offset = mu + std * eps
        else:
            offset = mu
        gate = self._format_variational_gate(gate)
        if gate is not None:
            offset = gate * offset
        return base + offset

    def variational_language_kl(self):
        if not self.use_variational_language:
            return None
        self.ensure_variational_language_parameters()
        prior_std = max(float(self.variational_language_prior_std), 1e-12)
        mu = self._language_feature_delta_mu
        log_sigma = self._language_feature_log_sigma.clamp(
            self.variational_language_log_sigma_min,
            self.variational_language_log_sigma_max,
        )
        sigma_rel = torch.exp(log_sigma)
        mean_rel = mu / prior_std
        kl = -log_sigma + 0.5 * (sigma_rel.pow(2) + mean_rel.pow(2) - 1.0)
        return kl.sum(dim=-1).mean()

    def variational_language_stats(self, gate=None):
        if not self.use_variational_language:
            return {}
        self.ensure_variational_language_parameters()
        with torch.no_grad():
            prior_std = float(self.variational_language_prior_std)
            log_sigma = self._language_feature_log_sigma.clamp(
                self.variational_language_log_sigma_min,
                self.variational_language_log_sigma_max,
            )
            sigma = prior_std * torch.exp(log_sigma)
            delta_mu = self._language_feature_delta_mu
            gate = self._format_variational_gate(gate)
            if gate is None:
                posterior_delta = delta_mu
                active = torch.ones(
                    delta_mu.shape[0], device=delta_mu.device, dtype=torch.bool
                )
            else:
                posterior_delta = gate * delta_mu
                active = gate.squeeze(-1) > 0
            inactive = ~active
            mu_norm = delta_mu.norm(dim=-1)

            def _mean_or_none(values):
                if values.numel() == 0:
                    return None
                return values.mean()

            posterior_mean = self._language_feature + posterior_delta
            return {
                "mu_mean": delta_mu.mean(),
                "mu_abs_mean": delta_mu.abs().mean(),
                "mu_norm_mean": mu_norm.mean(),
                "base_norm_mean": self._language_feature.norm(dim=-1).mean(),
                "posterior_mean_norm_mean": posterior_mean.norm(dim=-1).mean(),
                "sigma_mean": sigma.mean(),
                "sigma_norm_mean": sigma.norm(dim=-1).mean(),
                "log_sigma_mean": log_sigma.mean(),
                "mean_shift_norm_mean": posterior_delta.norm(dim=-1).mean(),
                "active_gate_ratio": active.float().mean(),
                "active_mu_norm_mean": _mean_or_none(mu_norm[active]),
                "inactive_mu_norm_mean": _mean_or_none(mu_norm[inactive]),
                "active_sigma_mean": _mean_or_none(sigma[active]),
                "inactive_sigma_mean": _mean_or_none(sigma[inactive]),
            }

    @property
    def get_text_language_feature(self):
        if self.text_language_feature is not None:
            return self.text_language_feature
        else:
            raise ValueError('没有设置text language feature')
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if training_args.include_feature:
            if self._language_feature is None or self._language_feature.shape[0] != self._xyz.shape[0]:
                language_feature = torch.zeros((self._xyz.shape[0], 16), device="cuda")
                self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
            if self.use_variational_language:
                self.ensure_variational_language_parameters()
                self._language_feature.requires_grad_(True)
                offset_lr_scale = float(getattr(
                    training_args,
                    "variational_language_offset_lr_scale",
                    0.1,
                ))
                default_offset_lr = training_args.language_feature_lr * offset_lr_scale
                mu_lr = float(getattr(training_args, "variational_language_mu_lr", 0.0))
                sigma_lr = float(getattr(training_args, "variational_language_sigma_lr", 0.0))
                if mu_lr <= 0.0:
                    mu_lr = default_offset_lr
                if sigma_lr <= 0.0:
                    sigma_lr = default_offset_lr
                l = [{
                    'params': [self._language_feature],
                    'lr': training_args.language_feature_lr,
                    "name": "language_feature",
                }, {
                    'params': [self._language_feature_delta_mu],
                    'lr': mu_lr,
                    'name': "language_feature_delta_mu",
                }]
                l.append({
                    'params': [self._language_feature_log_sigma],
                    'lr': sigma_lr,
                    'name': "language_feature_log_sigma",
                })
            else:
                l = [{
                    'params': [self._language_feature],
                    'lr': training_args.language_feature_lr,
                    "name": "language_feature",
                }]
            l.extend([
                 {'params': self.mlp1.parameters(), 'lr': training_args.mlp_lr, "name": "mlp1"},
                 {'params': self.mlp2.parameters(), 'lr': training_args.mlp_lr, "name": "mlp2"},
                 {'params': self.mlp3.parameters(), 'lr': training_args.mlp_lr, "name": "mlp3"},
                 {'params': self.cross_attention.parameters(), 'lr': training_args.mlp_lr, "name": "cross_attention"},
            ])
            if (not self.use_variational_language) and self.present_head is not None:
                l.append({
                    'params': self.present_head.parameters(),
                    'lr': training_args.mlp_lr,
                    'name': "present_head",
                })
            if (not self.use_variational_language) and self.gaussian_attr_conv_head is not None:
                l.append({
                    'params': self.gaussian_attr_conv_head.parameters(),
                    'lr': training_args.mlp_lr,
                    'name': "gaussian_attr_conv_head",
                })
            if (not self.use_variational_language) and self.sigma_head is not None:
                l.append({
                    'params': self.sigma_head.parameters(),
                    'lr': training_args.mlp_lr,
                    'name': "sigma_head",
                })
            if (not self.use_variational_language) and self.refer_uncertainty_head is not None:
                l.append({
                    'params': self.refer_uncertainty_head.parameters(),
                    'lr': training_args.mlp_lr,
                    'name': "refer_uncertainty_head",
                })
            self._xyz.requires_grad_(False)
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._opacity.requires_grad_(False)
        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # l.append('language_feature')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        print(self._xyz.shape[0])
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
