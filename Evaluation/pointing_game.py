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
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from pytorch_grad_cam import GradCAMPlusPlus
from captum.attr import LayerLRP, LayerIntegratedGradients, LayerDeepLiftShap
from Evaluation.bounding_box_dataset import *

# ============================================================================
# Core metric
# ============================================================================

def pointing_game_single(
    heatmap: np.ndarray,            # (H, W) float, any scale
    box:     Tuple[int,int,int,int],# (x1, y1, x2, y2) pixel coords
) -> bool:
    """
    Returns True if the heatmap peak falls inside the bounding box.
    Returns False if the heatmap is all-zero (degenerate case).
    """
    if heatmap.max() <= 0:
        return False
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    x1, y1, x2, y2 = box
    return (x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)


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
# Heatmap extraction utilities
# ============================================================================
def _paf_to_numpy_hw(
    p:    torch.Tensor,    # PAF distribution — any shape
    H:    int,
    W:    int,
) -> np.ndarray:
    """
    Convert PAF distribution to (H,W) numpy array.

    Rules:
    - Take MAX over channels (not sum) — preserves peak location
    - Use nearest or bilinear interpolation (not bicubic) — no ringing
    - Normalise to sum=1 — preserves probability interpretation for MiB
    - Never clip to [0,1] via max — preserves relative magnitudes
    """
    t = p.detach().cpu().float()

    # Remove batch dim
    if t.dim() == 4:
        t = t[0]                              # (C, h, w)

    if t.dim() == 3:
        # Take max over channels — preserves the strongest spatial signal
        # Sum diffuses peaks across channels with different spatial patterns
        t = t.abs().amax(dim=0)              # (h, w)
    elif t.dim() == 2:
        t = t.abs()                          # already (h, w)
    elif t.dim() == 1:
        return np.zeros((H, W))             # flat — no spatial info

    # t is now (h, w)
    h, w = t.shape
    if h != H or w != W:
        t = F.interpolate(
            t.unsqueeze(0).unsqueeze(0),     # (1, 1, h, w)
            size    = (H, W),
            mode    = 'bilinear',            # bilinear — no ringing
            align_corners = False,
        ).squeeze()                          # (H, W)

    arr = t.numpy().astype(np.float32)
    arr = np.clip(arr, 0, None)             # PAF is non-negative by design

    # Normalise to sum=1 — preserves probability interpretation
    # Do NOT divide by max — that destroys MiB
    s = arr.sum()
    if s > 0:
        arr = arr / s

    return arr


def _to_numpy_hw(
    attr: torch.Tensor,    # captum attribution — can be signed
    H:    int,
    W:    int,
) -> np.ndarray:
    """
    Convert baseline attribution (GradCAM, LRP, IG, DeepSHAP) to (H,W).

    Rules:
    - Sum over channels after abs — standard for gradient methods
    - Bilinear interpolation
    - Normalise to [0,1] via max — standard for pointing game
    """
    t = attr.detach().cpu().float()

    if t.dim() == 4:
        t = t[0]                             # (C, h, w)
    if t.dim() == 3:
        t = t.abs().sum(dim=0)              # (h, w) — sum ok for gradients
    elif t.dim() == 2:
        t = t.abs()
    elif t.dim() == 1:
        return np.zeros((H, W))

    h, w = t.shape
    if h != H or w != W:
        t = F.interpolate(
            t.unsqueeze(0).unsqueeze(0),
            size          = (H, W),
            mode          = 'bilinear',
            align_corners = False,
        ).squeeze()

    arr = t.numpy().astype(np.float32)
    arr = np.clip(arr, 0, None)

    # Normalise to [0,1] for pointing game peak detection
    if arr.max() > 0:
        arr = arr / arr.max()

    return arr

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
    hook_manager,
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

    input_layer = hook_manager.graph_info['backward_order'][-1]
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

            #hook_manager.run_forward(x)

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
    hook_manager,
    dataloader,
    device:       torch.device,
    paf_modes:    List[tuple],
    target_layer_name: str = 'features_28', #'layer4_1_conv2',
    num_samples:  Optional[int] = None,
    use_baselines: bool = True,
    thresholds:   List[float] = [0.5, 0.6, 0.7],
    print_every:  int = 30,
) -> Dict[str, Dict]:

    model.eval()

    # Accumulators
    hits        = defaultdict(int)
    total       = defaultdict(int)
    mass_scores = defaultdict(list)   # PAF only — continuous [0,1]
    th_hits     = {t: defaultdict(int) for t in thresholds}

    module_map   = hook_manager.graph_info['module_map']
    target_layer = module_map.get(target_layer_name)
    input_layer  = hook_manager.graph_info['backward_order'][-1]
    PAF_EVAL_LAYER = 'conv1' 
    if target_layer is None and use_baselines:
        raise ValueError(
            f"target_layer_name '{target_layer_name}' not found. "
            f"Available: {list(module_map.keys())[:10]}"
        )

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

            #hook_manager.run_forward(x)

            # --------------------------------------------------------
            # PAF
            # --------------------------------------------------------
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
                        print(f"PAF failed at sample {n_processed}: {e}")
                        n_processed += 1
                        continue

            for mode_key in distributions:
                mode_name = _mode_key_to_str(mode_key)
                store     = distributions[mode_key]
                p         = store.get(input_layer)

                if p is None:
                    continue

                
                heatmap = _to_numpy_hw(
                    p[0] if p.dim() == 4 else p, H, W
                )

                '''
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                raw  = x[0].permute(1,2,0).numpy()   # (H, W, 3)
                raw  = np.clip(raw * std + mean, 0, 1)    # denormalise

                #hmap=p.squeeze(0).sum(dim=1)
                #debug_box_alignment(heatmap,raw,box,mode_name,"PAF-output/debug"+mode_name+".png")
                
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from PIL import Image

                dataset = dataloader.dataset

                # Check box on TRANSFORMED image (not original)
                img_path, stem, class_idx = dataset.samples[0]
                img       = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img.size
                _, box_raw = dataset.box_map[stem]

                # Apply same transform as dataset
                transform = dataset.transform
                img_t     = transform(img)   # (3, 224, 224)

                # Transform box correctly
                box_transformed = transform_box_resize_centercrop(
                    box_raw, orig_w, orig_h
                )

                # Denormalise for display
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img_display = img_t.permute(1,2,0).numpy() * std + mean
                img_display = np.clip(img_display, 0, 1)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Original with raw box
                axes[0].imshow(img)
                x1, y1, x2, y2 = box_raw
                axes[0].add_patch(patches.Rectangle(
                    (x1,y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                ))
                axes[0].set_title(f'Original {orig_w}×{orig_h}\nbox raw: {box_raw}')
                axes[0].axis('off')

                # Transformed with corrected box
                axes[1].imshow(img_display)
                x1, y1, x2, y2 = box_transformed
                axes[1].add_patch(patches.Rectangle(
                    (x1,y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                ))
                axes[1].set_title(f'Transformed 224×224\nbox transformed: {box_transformed}')
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig('PAF-output/box_check_transformed.png', dpi=150, bbox_inches='tight')
                plt.show()
                '''

                # Re-normalise after interpolation
                '''
                s = heatmap.sum()
                if s > 0:
                    heatmap = heatmap / s
                '''

                # Standard pointing game
                pg_hit = pointing_game_single(heatmap, box)
                hits[f'PAF_{mode_name}']  += int(pg_hit)
                total[f'PAF_{mode_name}'] += 1

                #heatmap_clean = preprocess_heatmap_for_evaluation(
                #    heatmap,
                #    clip_low  = 1.0,
                #    clip_high = 99.0,
                #    renorm    = True,
                #)
                # Mass-in-box metrics (PAF only)
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

                # GradCAM++
                try:
                    cam = GradCAMPlusPlus(model=model,
                                         target_layers=[target_layer])
                    h   = cam(input_tensor=x)[0]
                    h   = np.clip(h, 0, None)
                    if h.max() > 0:
                        h = h / h.max()
                    hit = pointing_game_single(h, box)
                    hits['GradCAM++']  += int(hit)
                    total['GradCAM++'] += 1
                except Exception as e:
                    print(f"GradCAM++ failed: {e}")

                # LRP
                try:
                    llrp = LayerLRP(model, target_layer)
                    attr = llrp.attribute(x, target=label)
                    h    = _to_numpy_hw(attr, H, W)
                    hit  = pointing_game_single(h, box)
                    hits['LRP']  += int(hit)
                    total['LRP'] += 1
                except Exception as e:
                    print(f"LRP failed: {e}")

                # DeepSHAP
                try:
                    shap_model  = _make_shap_safe(model)
                    shap_layer  = _find_layer_in_copy(
                        model, shap_model, target_layer
                    )
                    ldls     = LayerDeepLiftShap(shap_model, shap_layer)
                    baseline = torch.zeros(
                        5, *x.shape[1:], device=x.device
                    )
                    attr = ldls.attribute(
                        x, baselines=baseline, target=label
                    )
                    h   = _to_numpy_hw(attr, H, W)
                    hit = pointing_game_single(h, box)
                    hits['DeepSHAP']  += int(hit)
                    total['DeepSHAP'] += 1
                except Exception as e:
                    print(f"DeepSHAP failed: {e}")

                # IG
                try:
                    lig  = LayerIntegratedGradients(model, target_layer)
                    attr = lig.attribute(
                        x,
                        baselines          = torch.zeros_like(x),
                        target             = label,
                        n_steps            = 50,
                        internal_batch_size= 1,
                    )
                    h   = _to_numpy_hw(attr, H, W)
                    hit = pointing_game_single(h, box)
                    hits['IG']  += int(hit)
                    total['IG'] += 1
                except Exception as e:
                    print(f"IG failed: {e}")

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

    # Final results
    results = _aggregate_results(
        hits, total, mass_scores, th_hits, thresholds
    )
    _print_results(results, thresholds)
    return results


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

'''
def evaluate_pointing_game(
    paf_class,                      # PAF class (not instance) for construction
    model:        nn.Module,
    hook_manager,                   # PAFHookManager instance
    dataloader,                     # yields (images, labels, boxes)
    device:       torch.device,
    paf_modes:    List[tuple],      # list of (mode, kwargs) pairs
    target_layer_name: str = 'layer4_1_conv2',  # for baselines
    num_samples:  Optional[int] = None,
    use_baselines: bool = True,
) -> Dict[str, Dict]:
    """
    Run Pointing Game evaluation for PAF (all modes) and baseline methods.

    Parameters
    ----------
    paf_class         : PAF class — used to construct PAF per image
    model             : nn.Module in eval mode
    hook_manager      : PAFHookManager — reused across images
    dataloader        : yields (images, labels, boxes)
                        boxes: (B, 4) tensor, pixel coords (x1,y1,x2,y2)
    device            : torch device
    paf_modes         : list of (ScoringMode, dict) pairs
    target_layer_name : layer name for baseline layer-wise attribution
    num_samples       : max images to evaluate (None = full dataset)
    use_baselines     : whether to run GradCAM, LRP, DeepSHAP, IG

    Returns
    -------
    results : {method_name: {'accuracy': float, 'hits': int, 'total': int}}
    """
    model.eval()

    # Accumulate hits/totals per method
    hits  = defaultdict(int)
    total = defaultdict(int)

    # Get baseline target layer module
    module_map   = hook_manager.graph_info['module_map']
    target_layer = module_map.get(target_layer_name)

    if target_layer is None and use_baselines:
        raise ValueError(
            f"target_layer_name '{target_layer_name}' not found in module_map. "
            f"Available: {list(module_map.keys())[:10]}"
        )

    # Find input layer (last in backward order = first layer of network)
    input_layer = hook_manager.graph_info['backward_order'][-1]

    n_processed = 0

    for batch_idx, batch in enumerate(tqdm(dataloader,
                                           desc='Pointing Game')):
        if num_samples is not None and n_processed >= num_samples:
            break

        # Unpack batch — support (img, label, box) and (img, label, box, *extras)
        images, labels, boxes = batch[0], batch[1], batch[2]
        images = images.to(device)
        labels = labels.to(device)

        # boxes: (B, 4) as (x1, y1, x2, y2) in pixel coords
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy().astype(int)
        B, C, H, W = images.shape

        # ----------------------------------------------------------------
        # PAF — one construction per image (PAF is per-sample)
        # ----------------------------------------------------------------
        for sample_idx in range(B):
            if num_samples is not None and n_processed >= num_samples:
                break

            x      = images[sample_idx:sample_idx+1]   # (1,C,H,W)
            label  = labels[sample_idx].item()
            box    = tuple(boxes[sample_idx])           # (x1,y1,x2,y2)

            # Run forward pass through hook manager
            #hook_manager.run_forward(x)
            #activations = hook_manager.activations
            #weights     = _extract_weights(model)

            try:
                paf_instance = paf_class(
                    model        = model,
                    hook_manager = hook_manager,
                    modes        = paf_modes,
                    x            = x,
                    target_class = label,
                    true_class   = label,
                )
                distributions = paf_instance.distributions
            except Exception as e:
                print(f"PAF failed at sample {n_processed}: {e}")
                n_processed += 1
                continue

            # Pointing game for each PAF mode
            for mode_key in distributions:
                mode_name = _mode_key_to_str(mode_key)
                heatmap   = _paf_input_heatmap(
                    distributions, mode_key, input_layer, H, W
                )
                hit = pointing_game_single(heatmap, box)
                hits[f'PAF_{mode_name}']  += int(hit)
                total[f'PAF_{mode_name}'] += 1

            # ----------------------------------------------------------------
            # Baselines
            # ----------------------------------------------------------------
            if use_baselines:

                # GradCAM++
                try:
                    cam  = GradCAMPlusPlus(model=model,
                                           target_layers=[target_layer])
                    h    = cam(input_tensor=x)[0]   # (H,W) already upsampled
                    h    = np.clip(h, 0, None)
                    if h.max() > 0:
                        h = h / h.max()
                    hit  = pointing_game_single(h, box)
                    hits['GradCAM++']  += int(hit)
                    total['GradCAM++'] += 1
                except Exception as e:
                    print(f"GradCAM++ failed: {e}")

                # LRP
                try:
                    llrp = LayerLRP(model, target_layer)
                    attr = llrp.attribute(x, target=label)
                    h    = _to_numpy_hw(attr, H, W)
                    hit  = pointing_game_single(h, box)
                    hits['LRP']  += int(hit)
                    total['LRP'] += 1
                except Exception as e:
                    print(f"LRP failed: {e}")

                # DeepSHAP
                try:
                    shap_model   = _make_shap_safe(model)
                    shap_layer   = _find_layer_in_copy(
                        model, shap_model, target_layer
                    )
                    ldls = LayerDeepLiftShap(shap_model, shap_layer)
                    baseline = torch.zeros(5, *x.shape[1:], device=x.device)

                    attr = ldls.attribute(
                        x,
                        baselines = baseline,
                        target    = label,
                    )
                    h   = _to_numpy_hw(attr, H, W)
                    hit = pointing_game_single(h, box)
                    hits['DeepSHAP']  += int(hit)
                    total['DeepSHAP'] += 1
                except Exception as e:
                    print(f"DeepSHAP failed: {e}")

                # IG
                try:
                    lig  = LayerIntegratedGradients(model, target_layer)
                    attr = lig.attribute(
                        x,
                        baselines          = torch.zeros_like(x),
                        target             = label,
                        n_steps            = 50,
                        internal_batch_size= 1,
                    )
                    h   = _to_numpy_hw(attr, H, W)
                    hit = pointing_game_single(h, box)
                    hits['IG']  += int(hit)
                    total['IG'] += 1
                except Exception as e:
                    print(f"IG failed: {e}")

            n_processed += 1

        # Progress log per batch
        if (batch_idx + 1) % 10 == 0:
            _print_current(hits, total, n_processed)

    # ----------------------------------------------------------------
    # Aggregate results
    # ----------------------------------------------------------------
    results = {}
    for method in set(list(hits.keys()) + list(total.keys())):
        h = hits.get(method, 0)
        t = total.get(method, 0)
        results[method] = {
            'accuracy': h / t * 100 if t > 0 else 0.0,
            'hits'    : h,
            'total'   : t,
        }

    _print_results(results)
    return results
'''

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


def _make_shap_safe(model: nn.Module) -> nn.Module:
    """Create a deepcopy with non-inplace ReLUs for DeepSHAP."""
    import copy, types
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
    except ImportError:
        return copy.deepcopy(model)

    shap_model = copy.deepcopy(model)

    def replace_relus(m):
        for name, child in m.named_children():
            if isinstance(child, nn.ReLU):
                setattr(m, name, nn.ReLU(inplace=False))
            else:
                replace_relus(child)

    replace_relus(shap_model)

    def bb_fwd(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)

    def btn_fwd(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu3(out + identity)

    for m in shap_model.modules():
        if isinstance(m, BasicBlock):
            m.relu1 = nn.ReLU(inplace=False)
            m.relu2 = nn.ReLU(inplace=False)
            m.forward = types.MethodType(bb_fwd, m)
        elif isinstance(m, Bottleneck):
            m.relu1 = nn.ReLU(inplace=False)
            m.relu2 = nn.ReLU(inplace=False)
            m.relu3 = nn.ReLU(inplace=False)
            m.forward = types.MethodType(btn_fwd, m)

    return shap_model


def _find_layer_in_copy(
    original: nn.Module,
    copy:     nn.Module,
    target:   nn.Module,
) -> nn.Module:
    """Find the equivalent of target layer in a deepcopy of the model."""
    for name, mod in original.named_modules():
        if mod is target:
            return dict(copy.named_modules())[name]
    raise ValueError("target layer not found in original model")



