"""Build the top-k Gaussian tokens used by the UC present classifier."""

import torch


def gaussian_attr_dim(sh_degree, feature_dim=128):
    """Dimension of [g, xyz, scale, rotation, opacity, SH, uncertainty]."""
    color_dim = 3 * (int(sh_degree) + 1) ** 2
    return int(feature_dim) + 3 + 3 + 4 + 1 + color_dim + 1


def normalize_scene_xyz(xyz):
    min_vals = xyz.min(dim=0).values
    max_vals = xyz.max(dim=0).values
    denom = (max_vals - min_vals).clamp_min(1e-6)
    return 2.0 * (xyz - min_vals) / denom - 1.0


def build_gaussian_attr_tokens(pc, g, indices):
    """Concatenate UC with the score-ranked top-k Gaussian evidence tokens.

    For each selected Gaussian i, the token is:
        [g_i, normalized_xyz_i, scale_i, rotation_i, opacity_i, SH_i, uc_i]

    Args:
        pc: GaussianModel with ``external_gaussian_uncertainty`` already loaded.
        g: Query-conditioned per-Gaussian features, shape [N, 128].
        indices: Top-k evidence Gaussian indices from the ReferSplat score path.

    Returns:
        attr_topk: [K, attr_dim] tensor for GaussianAttrConvPoolFormerHead.
        uc_topk: [K] uncertainty values, returned only for logging.
    """
    uc = getattr(pc, "external_gaussian_uncertainty", None)
    if uc is None:
        raise ValueError(
            "--use_gaussian_attr_conv_head requires external Gaussian uncertainty "
            "to be loaded"
        )
    uc = uc.to(device=g.device, dtype=g.dtype).reshape(-1)
    if uc.shape[0] != g.shape[0]:
        raise ValueError(
            f"External uncertainty length mismatch in renderer: got {uc.shape[0]}, "
            f"expected {g.shape[0]}"
        )

    idx = indices.to(device=g.device)
    xyz_norm = normalize_scene_xyz(pc.get_xyz).to(dtype=g.dtype)
    sh_flat = pc.get_features.reshape(pc.get_features.shape[0], -1).to(dtype=g.dtype)
    attr_topk = torch.cat([
        g[idx],
        xyz_norm[idx],
        pc._scaling[idx].to(dtype=g.dtype),
        pc.get_rotation[idx].to(dtype=g.dtype),
        pc.get_opacity[idx].to(dtype=g.dtype),
        sh_flat[idx],
        uc[idx].unsqueeze(-1),
    ], dim=-1)
    return attr_topk, uc[idx]
