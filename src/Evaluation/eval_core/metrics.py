"""
Evaluation Metrics
==================
Single source of truth for all PAF evaluation metrics.
No duplication across randomization_test, pointing_game, perturbation_test.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr,wilcoxon
from skimage.metrics import structural_similarity as ssim


def pointing_game_single(
    heatmap: np.ndarray,
    box:     Tuple[int,int,int,int],
) -> bool:
    if heatmap.max() <= 0:
        return False
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    x1, y1, x2, y2 = box
    return (x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)


def paf_box_evaluation(
    heatmap:    np.ndarray,
    box:        Tuple[int,int,int,int],
    thresholds: List[float] = [0.5, 0.6, 0.7],
) -> Dict:
    x1, y1, x2, y2 = box
    total_mass = heatmap.sum()
    if total_mass <= 0:
        return {'mass_in_box': 0.0, 'pointing_game': False,
                **{f'hit_{int(t*100)}': False for t in thresholds}}

    mass_in_box = heatmap[y1:y2+1, x1:x2+1].sum() / total_mass
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)

    result = {
        'mass_in_box'  : float(mass_in_box),
        'pointing_game': bool((x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)),
    }
    for t in thresholds:
        result[f'hit_{int(t*100)}'] = bool(mass_in_box >= t)
    return result


def spearman_correlation(
    h1: Optional[np.ndarray],
    h2: Optional[np.ndarray],
) -> Optional[float]:
    if h1 is None or h2 is None:
        return None
    h1_flat = h1.flatten()
    h2_flat = h2.flatten()
    if h1_flat.shape != h2_flat.shape:
        h2_flat = cv2.resize(h2, h1.shape[::-1]).flatten()
    corr, _ = spearmanr(h1_flat, h2_flat)
    return corr


def ssim_score(
    h1: Optional[np.ndarray],
    h2: Optional[np.ndarray],
) -> Optional[float]:
    if h1 is None or h2 is None:
        return None
    if h1.shape != h2.shape:
        h2 = cv2.resize(h2, (h1.shape[1], h1.shape[0]))
    h1_n = (h1 - h1.min()) / (h1.max() - h1.min() + 1e-8)
    h2_n = (h2 - h2.min()) / (h2.max() - h2.min() + 1e-8)
    return ssim(h1_n, h2_n, data_range=1.0)


def compute_similarity_scores(
    results_original: Dict[str, Optional[np.ndarray]],
    results_perturbed: Dict[str, Optional[np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """
    Compare heatmaps before and after perturbation.
    Returns {method: {'spearman': float, 'ssim': float}}.
    """
    scores = {}
    for name, h_orig in results_original.items():
        h_pert = results_perturbed.get(name)
        scores[name] = {
            'spearman': spearman_correlation(h_orig, h_pert),
            'ssim'    : ssim_score(h_orig, h_pert),
        }
    return scores

def _compute_audc(curves: np.ndarray) -> np.ndarray:
    """
    Exclude the first step (baseline — same for all methods) and
    last step ( same for all methods).
    Only steps 1-9/10 carry method-specific information.
    curves: (n_samples, n_steps) or (n_steps,) if only one sample.
    """
    curves = np.atleast_2d(curves) 
    curves_middle = curves[:, 1:-1]      # shape (n_samples, n_steps-2)
    x = np.linspace(0, 1, curves_middle.shape[1])
    return np.array([np.trapezoid(curve, x) for curve in curves_middle])


def _compute_auic(curves: np.ndarray) -> np.ndarray:
    """Same — exclude endpoints for insertion curves."""
    curves = np.atleast_2d(curves)    
    curves_middle = curves[:, 1:-1]
    x = np.linspace(0, 1, curves_middle.shape[1])
    return np.array([np.trapezoid(curve, x) for curve in curves_middle])

def _pad_incomplete_results(all_results, method_names):
    max_samples = max(len(all_results[n]['del']) for n in method_names)

    if max_samples == 0:
        raise RuntimeError("No successful samples for ANY method.")

    # Get n_steps from first method that has data
    n_steps = next(
        len(all_results[n]['del'][0])
        for n in method_names
        if all_results[n]['del']
    )

    for name in method_names:
        n_have    = len(all_results[name]['del'])
        n_missing = max_samples - n_have

        if n_missing == 0:
            continue

        print(f"WARNING: {name} has {n_have}/{max_samples} samples — "
              f"padding {n_missing} with {'mean' if n_have > 0 else 'neutral'} curve")

        if n_have == 0:
            # No data at all — use neutral 0.5 flat curve
            mean_del = [0.5] * n_steps
            mean_ins = [0.5] * n_steps
        else:
            mean_del = np.mean(all_results[name]['del'], axis=0).tolist()
            mean_ins = np.mean(all_results[name]['ins'], axis=0).tolist()

        for _ in range(n_missing):
            all_results[name]['del'].append(mean_del)
            all_results[name]['ins'].append(mean_ins)

    return all_results

def compute_aggregate_statistics(all_results, method_names, steps):
        """
        AUDC/AUIC, win rates, Wilcoxon p-values, Cohen's d.
        Reference method: first PAF mode found, or 'PAF' if present.
        """
        all_results = _pad_incomplete_results(all_results, method_names)
        del_matrix = {n: np.atleast_2d(np.array(all_results[n]['del'])) for n in method_names}
        ins_matrix = {n: np.atleast_2d(np.array(all_results[n]['ins'])) for n in method_names}
        audc = {n: _compute_audc(del_matrix[n]) for n in method_names}
        auic = {n: _compute_auic(ins_matrix[n]) for n in method_names}        

        audc_mat    = np.stack([audc[n] for n in method_names], axis=1)
        auic_mat    = np.stack([auic[n] for n in method_names], axis=1)
        del_winners = np.argmin(audc_mat, axis=1)
        ins_winners = np.argmax(auic_mat, axis=1)

        win_rate_del = {n: (del_winners == i).mean() * 100
                        for i, n in enumerate(method_names)}
        win_rate_ins = {n: (ins_winners == i).mean() * 100
                        for i, n in enumerate(method_names)}

        ref = 'PAF' if 'PAF' in method_names else \
              next((n for n in method_names if n.startswith('PAF_')), method_names[0])

        def cohens_d(a, b):
            diff = np.array(a) - np.array(b)
            return diff.mean() / (diff.std() + 1e-10)

        stats = {}
        for name in method_names:
            if name == ref:
                stats[name] = {
                    'audc_mean':    audc[name].mean(),
                    'audc_std':     audc[name].std(),
                    'auic_mean':    auic[name].mean(),
                    'auic_std':     auic[name].std(),
                    'win_rate_del': win_rate_del[name],
                    'win_rate_ins': win_rate_ins[name],
                    'p_del': None, 'p_ins': None,
                    'd_del': None, 'd_ins': None,
                }
                continue
            try:
                _, p_del = wilcoxon(audc[ref], audc[name])
                _, p_ins = wilcoxon(auic[ref], auic[name])
            except ValueError:
                p_del, p_ins = 1.0, 1.0

            stats[name] = {
                'audc_mean':    audc[name].mean(),
                'audc_std':     audc[name].std(),
                'auic_mean':    auic[name].mean(),
                'auic_std':     auic[name].std(),
                'win_rate_del': win_rate_del[name],
                'win_rate_ins': win_rate_ins[name],
                'p_del':        p_del,
                'p_ins':        p_ins,
                'd_del':        cohens_d(audc[ref], audc[name]),
                'd_ins':        cohens_d(auic[ref], auic[name]),
            }

        return stats, audc, auic, del_matrix, ins_matrix
