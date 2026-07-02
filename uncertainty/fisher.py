"""Core utilities for per-Gaussian Fisher uncertainty.

The experiment uses Fisher energy as a sensitivity score. A Gaussian with low
Fisher energy changes the rendered RGB less under small parameter movement, so
we convert it to high uncertainty by taking the negative log energy.
"""

import torch


DEFAULT_UNCERTAINTY_KEYS = (
    "uncertainty",
    "gaussian_uncertainty",
    "pup_uncertainty_rank01",
    "color_uncertainty_rank01",
)


def rank01_high(values):
    """Return rank-normalized scores in [0, 1], where larger values rank higher."""
    values = values.reshape(-1)
    if values.numel() <= 1:
        return torch.full_like(values, 0.5, dtype=torch.float32)
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def uncertainty_from_fisher_energy(energy, eps):
    """Convert Fisher energy to raw and rank-normalized uncertainty.

    Returns:
        log_energy: log(F_i + eps), used as the sensitivity diagnostic.
        uncertainty: -log(F_i + eps), where larger means more uncertain.
        uncertainty_rank01: rank-normalized uncertainty in [0, 1].
    """
    log_energy = torch.log(energy.clamp_min(float(eps)))
    uncertainty = -log_energy
    return log_energy, uncertainty, rank01_high(uncertainty)


def select_uncertainty_tensor(data, key=""):
    """Select a 1D uncertainty tensor from a saved .pt object."""
    if not isinstance(data, dict):
        return torch.as_tensor(data, dtype=torch.float32).reshape(-1), key or "<tensor>"

    if key:
        if key not in data:
            keys = ", ".join(sorted(str(k) for k in data.keys())[:20])
            raise KeyError(
                f"Uncertainty key {key!r} was not found. Available keys include: {keys}"
            )
        return torch.as_tensor(data[key], dtype=torch.float32).reshape(-1), key

    for candidate in DEFAULT_UNCERTAINTY_KEYS:
        if candidate in data:
            return torch.as_tensor(data[candidate], dtype=torch.float32).reshape(-1), candidate

    keys = ", ".join(sorted(str(k) for k in data.keys())[:20])
    raise KeyError(
        "No uncertainty key was provided and no default key was found. "
        f"Available keys include: {keys}"
    )


def load_uncertainty_tensor(path, key=""):
    """Load a per-Gaussian uncertainty vector from disk."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return select_uncertainty_tensor(data, key)
