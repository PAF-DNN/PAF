"""
Heatmap Utilities
=================
Convert PAF tensors and captum attributions to numpy heatmaps.
Single source of truth — no duplication across evaluation files.
"""

import torch
import torch.nn.functional as F
import numpy as np


def _paf_to_numpy_hw(p: torch.Tensor, H: int, W: int) -> np.ndarray:
    """PAF distribution → (H,W) numpy, sum=1. Uses amax over channels."""
    t = p.detach().cpu().float()
    if t.dim() == 4: t = t[0]
    if t.dim() == 3: t = t.sum(dim=0)
    elif t.dim() == 2: t = t.abs()
    elif t.dim() == 1: return np.zeros((H, W))

    if t.shape[0] != H or t.shape[1] != W:
        t = F.interpolate(t.unsqueeze(0).unsqueeze(0),
                          size=(H, W), mode='bilinear',
                          align_corners=False).squeeze()

    arr = t.numpy().astype(np.float32)
    arr = np.clip(arr, 0, None)
    s   = arr.sum()
    return arr / s if s > 0 else arr


def _to_numpy_hw(attr: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Captum attribution → (H,W) numpy, normalised to [0,1]."""
    t = attr.detach().cpu().float()
    if t.dim() == 4: t = t[0]
    if t.dim() == 3: t = t.abs().sum(dim=0)
    elif t.dim() == 2: t = t.abs()
    elif t.dim() == 1: return np.zeros((H, W))

    if t.shape[0] != H or t.shape[1] != W:
        t = F.interpolate(t.unsqueeze(0).unsqueeze(0),
                          size=(H, W), mode='bilinear',
                          align_corners=False).squeeze()

    arr = t.numpy().astype(np.float32)
    arr = np.clip(arr, 0, None)
    return arr / arr.max() if arr.max() > 0 else arr


def preprocess_heatmap_for_evaluation(
    heatmap:   np.ndarray,
    clip_low:  float = 1.0,
    clip_high: float = 99.0,
    renorm:    bool  = True,
) -> np.ndarray:
    """Percentile clip + renorm. Use for MiB only — not for PG peak."""
    h = np.clip(heatmap.copy().astype(np.float32), 0, None)
    if h.max() <= 0: return h
    lo, hi = np.percentile(h, clip_low), np.percentile(h, clip_high)
    h = np.clip(h, lo, hi) - np.clip(h, lo, hi).min()
    return h / h.sum() if renorm and h.sum() > 0 else h
