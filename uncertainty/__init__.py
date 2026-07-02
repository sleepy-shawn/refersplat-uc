"""Readable Fisher uncertainty + present-head components.

This package collects the uncertainty-specific pieces that were previously
spread across the renderer, model, and Fisher scripts.
"""

from .fisher import rank01_high, uncertainty_from_fisher_energy
from .gaussian_tokens import build_gaussian_attr_tokens, gaussian_attr_dim
from .present_classifier import GaussianAttrConvPoolFormerHead, PresentHead

__all__ = [
    "GaussianAttrConvPoolFormerHead",
    "PresentHead",
    "build_gaussian_attr_tokens",
    "gaussian_attr_dim",
    "rank01_high",
    "uncertainty_from_fisher_energy",
]
