"""
PAF Pointing Game Evaluation
=============================
Evaluates attribution methods using the Pointing Game metric.

The Pointing Game checks whether the maximum activation of a heatmap
falls inside the ground-truth bounding box of the target class.

Supports:
  - PAF (all scoring modes)
  - GradCAM++
  - LRP  (via captum LayerLRP)
  - DeepSHAP (via captum LayerDeepLiftShap)
  - IG  (via captum LayerIntegratedGradients)

Dataset requirement:
  The dataloader must yield (images, labels, boxes) where
  boxes is a tensor of shape (B, 4) in (x1, y1, x2, y2) pixel coordinates
  matching the image resolution.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from Evaluation.bounding_box_dataset import *
from Evaluation.eval_core.metrics import pointing_game_single, paf_box_evaluation
from Evaluation.eval_core.paf_runner import PAFRunner
from Evaluation.eval_core.baseline_runner import BaselineRunner
from Evaluation.eval_core.heatmap_utils import _paf_to_numpy_hw, _to_numpy_hw

# ============================================================================
# Core metric - now imported from Evaluation.core.metrics
# ============================================================================


def pointing_game_batch(
    heatmaps: List[np.ndarray],
    boxes:    List[Tuple[int,int,int,int]],
) -> Dict[str, float]:
    """
    Aggregate pointing game over a list of heatmaps and boxes.

    Returns
    -------
    dict with keys: accuracy, hits, total
    """
    assert len(heatmaps) == len(boxes), \
        f"Mismatch: {len(heatmaps)} heatmaps vs {len(boxes)} boxes"

    hits  = sum(pointing_game_single(h, b) for h, b in zip(heatmaps, boxes))
    total = len(heatmaps)
    return {
        'accuracy': hits / total * 100 if total > 0 else 0.0,
        'hits'    : hits,
        'total'   : total,
    }


# ============================================================================
# Heatmap extraction utilities - now imported from Evaluation.core.heatmap_utils
# ============================================================================



def _paf_input_heatmap(
    distributions: Dict,
    mode_key:      tuple,
    input_layer:   str,
    H: int, W: int,
    sample_idx: int = 0,
) -> np.ndarray:
    """
    Extract PAF attribution at the input layer and convert to (H,W).
    """
    store = distributions.get(mode_key, {})
    p     = store.get(input_layer)
    if p is None:
        return np.zeros((H, W))
    return _to_numpy_hw(p[sample_idx] if p.dim() == 4 else p, H, W)

'''
def paf_box_evaluation(
    heatmap:   np.ndarray,          # (H, W) — PAF distribution, sums to 1
    box:       Tuple[int,int,int,int],
    thresholds: List[float] = [0.5, 0.6, 0.7],
) -> Dict[str, float]:
    """
    Evaluate PAF attribution against a bounding box.

    Three metrics:
    1. mass_in_box   : fraction of total mass inside box
                       — the principled PAF metric
    2. pointing_game : standard peak-inside-box (for comparison)
    3. mass_hit_k    : whether mass_in_box >= threshold k
                       — binary version for accuracy reporting
    """
    x1, y1, x2, y2 = box

    total_mass  = heatmap.sum()
    if total_mass <= 0:
        return {'mass_in_box': 0.0, 'pointing_game': False,
                **{f'hit_{int(t*100)}': False for t in thresholds}}

    # Mass inside box
    box_region  = heatmap[y1:y2+1, x1:x2+1]
    mass_in_box = box_region.sum() / total_mass

    # Standard pointing game
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    peak_hit = (x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)

    result = {
        'mass_in_box'  : float(mass_in_box),
        'pointing_game': bool(peak_hit),
    }
    for t in thresholds:
        result[f'hit_{int(t*100)}'] = bool(mass_in_box >= t)

    return result
    

def evaluate_paf_box_mass(
    paf_class,
    model,
    graph_manager,
    dataloader,
    device,
    paf_modes,
    num_samples:  int = 1000,
    thresholds:   List[float] = [0.5, 0.6, 0.7],
) -> Dict:
    """
    Run box mass evaluation for PAF.
    Reports both mass-in-box (principled) and pointing game (comparable).
    """
    from collections import defaultdict
    import numpy as np
    from tqdm import tqdm

    # Accumulators per PAF mode
    mass_scores = defaultdict(list)   # continuous [0,1] per image
    pg_hits     = defaultdict(int)    # pointing game hits
    th_hits     = {
        t: defaultdict(int) for t in thresholds
    }
    total       = defaultdict(int)

    input_layer = graph_manager.graph_info['backward_order'][-1]
    n_processed = 0

    for batch in tqdm(dataloader, desc='Box Mass Evaluation'):
        if n_processed >= num_samples:
            break

        images, labels, boxes = batch[0], batch[1], batch[2]
        images = images.to(device)
        labels = labels.to(device)
        B, C, H, W = images.shape

        for si in range(B):
            if n_processed >= num_samples:
                break

            x     = images[si:si+1]
            label = labels[si].item()
            box   = tuple(boxes[si].cpu().numpy().astype(int))

            #graph_manager.run_forward(x)

            try:
                paf_instance  = paf_class(
                    model        = model,
                    hook_manager = hook_manager,
                    modes        = paf_modes,
                    x            = x,
                    target_class = label,
                    true_class   = label,
                )
                distributions = paf_instance.distributions
            except Exception as e:
                print(f"PAF failed: {e}")
                n_processed += 1
                continue

            for mode_key in distributions:
                mode_name = _mode_key_to_str(mode_key)
                store     = distributions[mode_key]
                p         = store.get(input_layer)

                if p is None:
                    continue

                # Convert to (H,W) numpy — already a probability distribution
                heatmap = _paf_to_numpy_hw(
                    p[0] if p.dim() == 4 else p, H, W
                )
                # Re-normalise to sum=1 after any interpolation artifacts
               # s = heatmap.sum()
               # if s > 0:
               #     heatmap = heatmap / s

                metrics = paf_box_evaluation(heatmap, box, thresholds)

                mass_scores[mode_name].append(metrics['mass_in_box'])
                pg_hits[mode_name]    += int(metrics['pointing_game'])
                for t in thresholds:
                    th_hits[t][mode_name] += int(metrics[f'hit_{int(t*100)}'])
                total[mode_name] += 1

            n_processed += 1

    # ----------------------------------------------------------------
    # Aggregate and print
    # ----------------------------------------------------------------
    results = {}
    print("\n" + "=" * 75)
    print(f"{'Method':<35} {'MassInBox':>10} {'PG%':>7} "
          + "  ".join(f"H{int(t*100)}%" for t in thresholds))
    print("-" * 75)

    for mode_name in sorted(mass_scores.keys()):
        n     = total[mode_name]
        scores= np.array(mass_scores[mode_name])
        mean  = scores.mean()
        ci    = 1.96 * scores.std() / np.sqrt(n)
        pg    = pg_hits[mode_name] / n * 100
        th_accs = [th_hits[t][mode_name] / n * 100 for t in thresholds]

        marker = ' ★' if 'norm_power' in mode_name else ''
        print(
            f"  {mode_name+marker:<33} "
            f"{mean:.4f}±{ci:.4f}  "
            f"{pg:>6.1f}%  "
            + "  ".join(f"{a:>6.1f}%" for a in th_accs)
        )

        results[mode_name] = {
            'mass_in_box_mean' : float(mean),
            'mass_in_box_ci95' : float(ci),
            'mass_in_box_scores': scores.tolist(),
            'pointing_game_acc': pg,
            **{f'hit_{int(t*100)}_acc': th_hits[t][mode_name]/n*100
               for t in thresholds},
            'n': n,
        }

    print("=" * 75)
    print(f"\nNote: MassInBox = fraction of PAF probability mass inside "
          f"ground-truth box.")
    print(f"      Thresholds H50/H60/H70 = % images where MassInBox >= "
          f"50/60/70%.")
    print(f"      PG = standard pointing game (peak pixel inside box).")

    return results
    

def preprocess_heatmap_for_evaluation(
    heatmap:    np.ndarray,    # (H, W) raw PAF distribution
    clip_low:   float = 1.0,   # percentile — remove bottom noise
    clip_high:  float = 99.0,  # percentile — remove top outliers
    renorm:     bool  = True,  # re-normalise to sum=1 after clipping
) -> np.ndarray:
    """
    Remove outliers from heatmap before evaluation.

    Steps:
    1. Clip values outside [clip_low, clip_high] percentile range
    2. Zero out negative values (should not exist for PAF but safety)
    3. Re-normalise to sum=1 so MiB interpretation is preserved

    For PAF: use clip_low=1, clip_high=99
    For baselines: use clip_low=0, clip_high=99 (no lower clip needed
                   since we already take abs)

    Note: do NOT clip before pointing game peak detection — the peak
    should be found on the raw heatmap. Clip only for MiB computation.
    """
    h = heatmap.copy().astype(np.float32)
    h = np.clip(h, 0, None)   # remove any negatives

    if h.max() <= 0:
        return h

    lo = np.percentile(h, clip_low)
    hi = np.percentile(h, clip_high)

    h = np.clip(h, lo, hi)
    h = h - h.min()   # shift so minimum is 0 after clipping

    if renorm and h.sum() > 0:
        h = h / h.sum()   # re-normalise to sum=1

    return h
    '''

def debug_box_alignment(
    heatmap:   np.ndarray,    # (H, W) float, sum=1
    raw_img:   np.ndarray,    # (H, W, 3) float in [0,1]
    box:       tuple,         # (x1, y1, x2, y2)
    method:    str,
    save_path: str = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # ----------------------------------------------------------------
    # Fix shapes before anything else
    # ----------------------------------------------------------------

    # Fix raw_img: (3, H, W) → (H, W, 3)
    if raw_img.ndim == 3 and raw_img.shape[0] == 3:
        raw_img = raw_img.transpose(1, 2, 0)

    # Fix raw_img: (1, H, W, 3) or (1, 3, H, W) → (H, W, 3)
    if raw_img.ndim == 4:
        raw_img = raw_img.squeeze(0)
        if raw_img.shape[0] == 3:
            raw_img = raw_img.transpose(1, 2, 0)

    # Normalise raw_img to [0,1]
    raw_img = raw_img.astype(np.float32)
    if raw_img.max() > 1.0:
        raw_img = raw_img / 255.0
    raw_img = np.clip(raw_img, 0, 1)

    H, W = raw_img.shape[:2]

    # Fix heatmap: (C, H, W) → (H, W) via max
    if heatmap.ndim == 3:
        heatmap = heatmap.max(axis=0)

    # Fix heatmap: (1, H, W) → (H, W)
    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap.squeeze(0)

    # Fix heatmap: (1, C, H, W) → (H, W)
    if heatmap.ndim == 4:
        heatmap = heatmap.squeeze(0).max(axis=0)

    # Resize heatmap to match raw_img if needed
    if heatmap.shape != (H, W):
        heatmap = cv2.resize(
            heatmap.astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )

    # Normalise heatmap to sum=1
    heatmap = np.clip(heatmap, 0, None).astype(np.float32)
    s = heatmap.sum()
    if s > 0:
        heatmap = heatmap / s

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------
    x1, y1, x2, y2 = box
    # Clamp box to image bounds
    x1 = max(0, min(x1, W-1))
    x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1))
    y2 = max(0, min(y2, H-1))

    box_area    = (x2 - x1) * (y2 - y1)
    image_area  = H * W
    box_coverage= box_area / image_area * 100

    mass_in  = heatmap[y1:y2+1, x1:x2+1].sum()
    total    = heatmap.sum()
    mib      = mass_in / total if total > 0 else 0

    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pg_hit = (x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    def _add_box(ax):
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='red', facecolor='none',
        )
        ax.add_patch(rect)

    # Panel 1: raw image + box
    axes[0].imshow(raw_img)
    _add_box(axes[0])
    axes[0].set_title('Input + GT Box\n'
                       f'Box coverage: {box_coverage:.1f}% of image')
    axes[0].axis('off')

    # Panel 2: heatmap + box + peak marker
    im = axes[1].imshow(heatmap, cmap='inferno')
    _add_box(axes[1])
    color = 'lime' if pg_hit else 'red'
    axes[1].plot(
        peak_x, peak_y, '+',
        color=color, markersize=15, markeredgewidth=3,
        label=f'Peak ({"HIT" if pg_hit else "MISS"})',
    )
    axes[1].legend(fontsize=9, loc='upper right')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f'{method}\n'
                       f'MiB={mib:.4f}  PG={"✓" if pg_hit else "✗"}')
    axes[1].axis('off')

    # Panel 3: blend overlay + box
    h_rgb   = plt.cm.inferno(heatmap)[:, :, :3]
    blended = np.clip(0.4 * raw_img + 0.6 * h_rgb, 0, 1)
    axes[2].imshow(blended)
    _add_box(axes[2])
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    fig.suptitle(
        f'Box: ({x1},{y1})→({x2},{y2})  '
        f'Coverage: {box_coverage:.1f}%  '
        f'MiB: {mib:.4f}  '
        f'Peak: ({peak_x},{peak_y})  '
        f'PG: {"HIT ✓" if pg_hit else "MISS ✗"}',
        fontsize=10, y=1.02,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()
# ============================================================================
# Main evaluation loop
# ============================================================================
def evaluate_pointing_game(
    paf_class,
    model:        nn.Module,
    graph_manager,
    dataloader,
    device:       torch.device,
    paf_modes:    List[tuple],
    target_layer_name: str = None,
    num_samples:  Optional[int] = None,
    use_baselines: bool = True,
    thresholds:   List[float] = [0.5, 0.6, 0.7],
    print_every:  int = 30,
) -> Dict[str, Dict]:

    model.eval()
    hits        = defaultdict(int)
    total       = defaultdict(int)
    mass_scores = defaultdict(list)
    th_hits     = {t: defaultdict(int) for t in thresholds}
    module_map   = graph_manager.graph_info.module_map
    #target_layer = module_map.get(target_layer_name)
    input_layer  = graph_manager.graph_info.backward_order[-1]

    # Only resolve target_layer if baselines are requested
    target_layer = None
    if use_baselines:
        if target_layer_name is None:
            # Auto-detect last conv layer
            target_layer = next(
                (m for m in reversed(list(module_map.values()))
                 if isinstance(m, nn.Conv2d)),
                None
            )
            if target_layer is None:
                print("WARNING: No Conv2d found — disabling baselines")
                use_baselines = False
        else:
            target_layer = module_map.get(target_layer_name)
            if target_layer is None:
                print(
                    f"WARNING: target_layer_name '{target_layer_name}' not found. "
                    f"Available: {list(module_map.keys())[:10]}\n"
                    f"Disabling baselines — PAF will still run."
                )
                use_baselines = False

    baseline_runner = BaselineRunner(model, target_layer) if use_baselines else None
    paf_runner = PAFRunner(model, graph_manager, paf_modes)
    n_processed = 0

    for batch_idx, batch in enumerate(tqdm(dataloader,
                                           desc='Pointing Game')):
        if num_samples is not None and n_processed >= num_samples:
            break

        images, labels, boxes = batch[0], batch[1], batch[2]
        images = images.to(device)
        labels = labels.to(device)
        B, C, H, W = images.shape

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy().astype(int)

        for si in range(B):
            if num_samples is not None and n_processed >= num_samples:
                break

            x     = images[si:si+1]
            label = labels[si].item()
            box   = tuple(boxes[si])

            # --------------------------------------------------------
            # PAF
            # --------------------------------------------------------
            try:
                heatmaps = paf_runner.get_heatmaps(
                    x, label, input_layer, H, W, true_class=label
                )
            except Exception as e:
                print(f"PAF failed at sample {n_processed}: {e}")
                n_processed += 1
                continue

            for mode_name, heatmap in heatmaps.items():
                if heatmap is None:
                    continue

                pg_hit = pointing_game_single(heatmap, box)
                hits[f'PAF_{mode_name}']  += int(pg_hit)
                total[f'PAF_{mode_name}'] += 1

                metrics = paf_box_evaluation(heatmap, box, thresholds)
                mass_scores[f'PAF_{mode_name}'].append(
                    metrics['mass_in_box']
                )
                for t in thresholds:
                    th_hits[t][f'PAF_{mode_name}'] += int(
                        metrics[f'hit_{int(t*100)}']
                    )

            # --------------------------------------------------------
            # Baselines — standard pointing game only
            # --------------------------------------------------------
            if use_baselines:
                baseline_results = baseline_runner.run(x, label, H, W)
                for method_name, heatmap in baseline_results.items():
                    if heatmap is None:
                        continue
                    hit = pointing_game_single(heatmap, box)
                    hits[method_name] += int(hit)
                    total[method_name] += 1

            n_processed += 1

            # --------------------------------------------------------
            # Progress report every print_every samples
            # --------------------------------------------------------
            if n_processed % print_every == 0:
                _print_current(
                    hits        = hits,
                    total       = total,
                    mass_scores = mass_scores,
                    th_hits     = th_hits,
                    thresholds  = thresholds,
                    n           = n_processed,
                )

    # Final clean-up
    paf_runner.cleanup()

    # Final results
    results = _aggregate_results(
        hits, total, mass_scores, th_hits, thresholds
    )
    _print_results(results, thresholds)
    return hits, total, mass_scores, th_hits


# ============================================================================
# Updated helpers
# ============================================================================

def _print_current(
    hits:        dict,
    total:       dict,
    mass_scores: dict,
    th_hits:     dict,
    thresholds:  List[float],
    n:           int,
) -> None:
    th_header = "  ".join(f"H{int(t*100)}%" for t in thresholds)
    print(f"\n  [{n} samples processed]")
    print(f"  {'Method':<38} {'PG%':>7} {'MiB':>8}  {th_header}")
    print("  " + "-" * 72)

    paf_methods  = sorted(k for k in total if k.startswith('PAF'))
    base_methods = sorted(k for k in total if not k.startswith('PAF'))

    for method in paf_methods + base_methods:
        t = total.get(method, 0)
        if t == 0:
            continue
        h  = hits.get(method, 0)
        pg = h / t * 100

        if method.startswith('PAF') and mass_scores.get(method):
            scores = np.array(mass_scores[method])
            mib    = f"{scores.mean():.4f}"
            th_str = "  ".join(
                f"{th_hits[thr].get(method, 0) / t * 100:>6.1f}%"
                for thr in thresholds
            )
        else:
            mib    = "   N/A "
            th_str = "  ".join("   N/A" for _ in thresholds)

        marker = ' ★' if 'norm_power' in method else ''
        print(f"    {method+marker:<36} {pg:>6.1f}%  "
              f"{mib:>8}  {th_str}")

def _aggregate_results(
    hits:        dict,
    total:       dict,
    mass_scores: dict,
    th_hits:     dict,
    thresholds:  List[float],
) -> Dict:
    results = {}
    all_methods = set(list(hits.keys()) + list(total.keys()))

    for method in all_methods:
        t = total.get(method, 0)
        h = hits.get(method, 0)
        r = {
            'pointing_game_acc': h / t * 100 if t > 0 else 0.0,
            'hits' : h,
            'total': t,
        }
        if method.startswith('PAF') and mass_scores.get(method):
            scores = np.array(mass_scores[method])
            ci     = 1.96 * scores.std() / np.sqrt(len(scores))
            r['mass_in_box_mean']   = float(scores.mean())
            r['mass_in_box_ci95']   = float(ci)
            r['mass_in_box_scores'] = scores.tolist()
            for thr in thresholds:
                r[f'hit_{int(thr*100)}_acc'] = \
                    th_hits[thr].get(method, 0) / t * 100 if t > 0 else 0.0
        results[method] = r

    return results


def _print_results(results: Dict, thresholds: List[float]) -> None:
    th_header = "  ".join(f"H{int(t*100)}%" for t in thresholds)
    print("\n" + "=" * 80)
    print(f"  {'Method':<35} {'PG%':>7} {'MiB mean':>10} "
          f"{'CI95':>8}  {th_header}")
    print("-" * 80)

    paf_methods  = sorted(k for k in results if k.startswith('PAF'))
    base_methods = sorted(k for k in results if not k.startswith('PAF'))

    for method in paf_methods + base_methods:
        r      = results[method]
        pg     = r['pointing_game_acc']
        marker = ' ★' if 'norm_power' in method else ''

        if method.startswith('PAF') and 'mass_in_box_mean' in r:
            mib    = f"{r['mass_in_box_mean']:.4f}"
            ci     = f"±{r['mass_in_box_ci95']:.4f}"
            th_str = "  ".join(
                f"{r.get(f'hit_{int(t*100)}_acc', 0):>6.1f}%"
                for t in thresholds
            )
        else:
            mib    = "   N/A  "
            ci     = "     N/A"
            th_str = "  ".join("  N/A " for _ in thresholds)

        print(f"  {method+marker:<35} {pg:>6.1f}%  "
              f"{mib:>8}  {ci:>8}  {th_str}")

    print("=" * 80)
    print(f"\n  PG   = Pointing Game (peak pixel in box)")
    print(f"  MiB  = Mass-in-Box (PAF only — fraction of Σ=1 inside box)")
    print(f"  H50/H60/H70 = % images with MiB ≥ 50/60/70%")
    print(f"  N/A  = metric not applicable for gradient-based methods\n")


# ============================================================================
# Helpers
# ============================================================================

def _extract_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract named weights from all modules."""
    weights = {}
    for name, module in model.named_modules():
        fx_name = name.replace('.', '_')
        if hasattr(module, 'weight') and module.weight is not None:
            weights[fx_name] = module.weight.data
    return weights


def _mode_key_to_str(mode_key: tuple) -> str:
    """Convert a PAF mode key tuple to a readable string."""
    parts = []
    for part in mode_key:
        if hasattr(part, 'value'):
            parts.append(str(part.value))
        elif isinstance(part, float):
            parts.append(f"t{part:.1f}")
        else:
            parts.append(str(part))
    return '_'.join(parts)

