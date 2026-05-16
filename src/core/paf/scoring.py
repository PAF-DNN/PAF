"""
Scoring modes for Probabilistic Activation Flow (PAF).

Distribute probability mass backward through the network.
"""

import torch
from enum import Enum
from typing import Callable, Dict, Any, Tuple


# =================================================================
# 1. Enum - type definition only
# =================================================================

class ScoringMode(Enum):
    """
    Controls how PAF computes the attribution signal s_jk
    used to distribute probability mass backward through the network.

    All modes produce non-negative scores suitable for normalisation.
    Sign information is always tracked separately from the original
    product sign(a_j * w_jk) regardless of scoring mode.
    """
    ABS          = "abs"           # |a * w|  — current default
    POWER        = "power"         # |a * w|^tau
    EXP_WEIGHT   = "exp_weight"    # |a| * exp(|w|)
    NORM         = "norm"          # |â * ŵ| where â=a/max(a), ŵ=w/max(w)
    NORM_POWER  = "norm_power"    # |â * ŵ|^tau — normalised + sharpened
    SIGNED_SPLIT = "signed_split"  # LRP-style
    SIGNED_FULL  = "signed_full"


# =================================================================
# 2. Parameter validation - simple, no class needed
# =================================================================

def validate_scoring_params(mode: ScoringMode, **kwargs) -> dict:
    """
    Validate and fill defaults for scoring parameters.
    Only real constraints: tau > 0, alpha >= 0, beta >= 0.
    """
    tau = kwargs.get('tau', 1.0)
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 0.0)

    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if mode == ScoringMode.SIGNED_SPLIT:
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be >= 0, got {alpha}, {beta}")

    return {'tau': tau, 'alpha': alpha, 'beta': beta}


# =================================================================
# 3. Private builders - one per mode, independently testable
# =================================================================

def _build_abs() -> Callable:
    """Absolute scoring: |a * w|"""
    def score_fn(a: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = a * w
        return z.abs(), z
    return score_fn


def _build_power(tau: float) -> Callable:
    """Power scoring: |a * w|^tau"""
    def score_fn(a: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = a * w
        return z.abs().pow(tau), z
    return score_fn


def _build_norm() -> Callable:
    """Normalised scoring: |â * ŵ| where â=a/max(a), ŵ=w/max(w)"""
    call_count = [0]
    def score_fn(a: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_max = a.abs().amax(dim=2, keepdim=True)
        w_max = w.abs().amax(dim=2, keepdim=True)
        
        # Prevent division by zero
        a_max = a_max.clamp(min=1e-12)
        w_max = w_max.clamp(min=1e-12)
        
        a_hat = a.abs() / a_max
        w_hat = w.abs() / w_max
        
        raw = a_hat * w_hat
        K      = raw.sum(dim=2, keepdim=True).clamp(min=1e-12)
        scores=raw/K
        z=a*w
        return scores, z
    score_fn.pre_normalised = True
    return score_fn


def _build_norm_power(tau: float) -> Callable:
    """Normalised power scoring: |â * ŵ|^tau"""
    def score_fn(a: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_max = a.abs().amax(dim=2, keepdim=True)
        w_max = w.abs().amax(dim=2, keepdim=True)
        
        # Prevent division by zero
        a_max = a_max.clamp(min=1e-12)
        w_max = w_max.clamp(min=1e-12)
        
        a_hat = a.abs() / a_max
        w_hat = w.abs() / w_max
        raw    = (a_hat * w_hat).pow(tau)
        K      = raw.sum(dim=2, keepdim=True).clamp(min=1e-12)
        scores = raw / K
        z      = a * w
        return scores, z
    score_fn.pre_normalised = True
    return score_fn


def _build_signed_split(alpha: float, beta: float, eps: float) -> Callable:
    """Signed split scoring with alpha/beta parameters."""
    def score_fn(a: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = a * w
        z_pos = z.clamp(min=0)
        z_neg = z.clamp(max=0).abs()
        
        K_pos = z_pos.sum(dim=2, keepdim=True).clamp(min=eps)
        K_neg = z_neg.sum(dim=2, keepdim=True).clamp(min=eps)
        
        pos_contrib = torch.where(
            K_pos <= eps,
            torch.zeros_like(z_pos),
            z_pos / K_pos.clamp(min=eps)
        )
        
        neg_contrib = torch.where(
            K_neg <= eps,
            torch.zeros_like(z_neg),
            z_neg / K_neg.clamp(min=eps)
        )
        
        scores = alpha * pos_contrib + beta * neg_contrib
        return scores, z
    return score_fn


def _build_signed_full() -> Callable:
    """Full signed attribution: a * w"""
    def score_fn(a: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = a * w
        return z, z
    return score_fn


# =================================================================
# 4. Public factory - single entry point
# =================================================================

_BUILDERS = {
    ScoringMode.ABS:          lambda p, eps: _build_abs(),
    ScoringMode.POWER:        lambda p, eps: _build_power(p['tau']),
    ScoringMode.NORM:         lambda p, eps: _build_norm(),
    ScoringMode.NORM_POWER:  lambda p, eps: _build_norm_power(p['tau']),
    ScoringMode.SIGNED_SPLIT: lambda p, eps: _build_signed_split(p['alpha'], p['beta'], eps),
    ScoringMode.SIGNED_FULL:  lambda p, eps: _build_signed_full(),
}


def build_scoring(
    mode: ScoringMode,
    tau: float = 1.0,
    eps: float = 1e-9,
    **kwargs
) -> Callable:
    """
    Build a score_fn closure for the given mode.
    score_fn(a, w) → (scores, z) where scores >= 0.
    """
    if mode not in _BUILDERS:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(_BUILDERS)}")

    params = validate_scoring_params(mode, tau=tau, **kwargs)
    return _BUILDERS[mode](params, eps)


# =================================================================
# 5. Mode key - hashable tuple for distributions dict
# =================================================================

def make_mode_key(mode: ScoringMode, tau: float = 1.0, **kwargs) -> tuple:
    """Hashable key for (mode, hyperparameters) — used as dict key in PAF."""
    if mode == ScoringMode.SIGNED_SPLIT:
        return (mode, tau, kwargs.get('alpha', 1.0), kwargs.get('beta', 0.0))
    return (mode, tau)
