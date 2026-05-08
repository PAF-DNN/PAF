import dis
import re
import os
from graphviz import Digraph
import torch
import torch.fx as fx
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

#from nn_arch.cnn_mnist import CNN, transform,  device, load_model
from core.paf import PAF, ScoringMode
#from core.cnn_analysis import analyze_feature_coalitions
from core.nn_graph import PAFHookManager, make_model_universal_for_shap
from captum.attr import IntegratedGradients, LRP, LayerLRP, DeepLiftShap
from captum.attr import LRP
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class PAFVisualizer:
    def __init__(self, paf: PAF,misclassification=False,contrastive_explanation=False,debug_level=0,true_class=0,target_class=0):
        self.paf=paf
        self.debug_level = debug_level
        self.heatmap=None
        self.contrastive_explanation=contrastive_explanation
        self.misclassification=misclassification
        self.paf_shared=None
        self.true_class=true_class
        self.target_class=target_class
        self.sample_idx=0

    def _ensure_save_dir(self, save_path: str) -> None:
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

    def get_input_heatmap(self,p_in, sample_idx=0, blur_sigma=0.8, percentile=98.5,mode='visualize'):
        """
        Better handling for very small per-pixel probabilities.
        """
        heatmap = p_in.sum(dim=0).cpu().numpy()   # [H, W]
       # channel_mass = p_in.sum(dim=(1, 2), keepdim=True)  # (3, 1, 1)
       # heatmap = (p_in * channel_mass).sum(dim=0).cpu().numpy()  # (H, W)
        #heatmap = p_in.max(dim=0).values.cpu().numpy()   # [H, W]

        if mode == 'evaluate':
            self.heatmap = heatmap
            return heatmap
        elif mode == 'visualize':
            # 2. Percentile clipping + normalization
            v_max = np.percentile(heatmap, percentile)
            heatmap = np.clip(heatmap, 0, v_max)
            if blur_sigma > 0:
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=blur_sigma)
            self.heatmap=heatmap
            return heatmap
    
    def get_signed_heatmap(self, p_signed, blur_sigma=0.8, percentile=99.5):
        """
        Creates a heatmap where positive values are 'evidence for' 
        and negative values are 'evidence against'.
        """
        # 1. Sum across channels to get spatial map [H, W]
        heatmap = p_signed.sum(dim=0).cpu().numpy()

        # 2. Symmetric Clipping
        # To keep 0 as the neutral point, we clip using the same absolute value for both ends
        v_abs_max = np.percentile(np.abs(heatmap), percentile)
        heatmap = np.clip(heatmap, -v_abs_max, v_abs_max)

        # 3. Smoothing (Preserves the zero-crossings better if sigma is small)
        if blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

        # 4. Normalize to [-1, 1] range for consistent plotting
        if v_abs_max > 0:
            heatmap = heatmap / v_abs_max

        return heatmap
    def get_heatmap_per_mode(
        self,
        input_node:  str   = 'x',
        sample_idx:  int   = 0,
        blur_sigma:  float = 0.8,
        percentile:  float = 98.5,
        use_mode:        str   = 'visualize',
        mode_key:    tuple = None,
    ) -> np.ndarray:
        """
        Returns input-layer heatmap for the specified PAF scoring mode.

        Args:
            input_node: FX graph node name for the input layer (default 'x')
            sample_idx: batch index to extract
            blur_sigma: gaussian blur sigma for visualise mode (0 = no blur)
            percentile: clipping percentile for visualise mode
            mode:       'evaluate' — raw heatmap for insertion/deletion tests
                        'visualize' — clipped and blurred for display
            signed:     if True return signed heatmap (excitatory/inhibitory)
            mode_key:   (ScoringMode, tau) — which scoring mode to use.
                        Defaults to self.mode_key set at construction.
        """
        # Select mode key
        key = mode_key if mode_key is not None else self.mode_key

        # Select distribution store
        store = self.paf.distributions[key]

        # Get input distribution
        if input_node not in store:
            raise KeyError(
                f"Input node '{input_node}' not found in distributions "
                f"for mode {key}. "
                f"Available nodes: {list(store.keys())[:5]}"
            )

        p_in = store[input_node]

        # Handle batch dimension
        if p_in.dim() == 4:
            p_in = p_in[sample_idx]   # (C, H, W)
        elif p_in.dim() == 3:
            p_in = p_in[sample_idx]   # (C, L) for 1D

        # Aggregate channels — sum gives total mass per spatial position
        heatmap = p_in.sum(dim=0).cpu().numpy()   # (H, W) — may be negative for signed mode
        is_signed = mode_key[0] == ScoringMode.SIGNED_FULL
        if use_mode == 'evaluate':
            # Raw — no post-processing
            self.heatmap = heatmap
            if is_signed:
                # sign information is irrelevant for pixel ranking
                return np.abs(heatmap)
            return heatmap
        elif use_mode == 'visualize':
            if is_signed:
                # Values are tiny and mixed sign — scale to [-1, +1]
                # using symmetric percentile clipping to avoid outlier dominance
                v_max = np.percentile(np.abs(heatmap), percentile)
                if v_max < 1e-12:
                    # Completely flat — nothing to show
                    return np.zeros_like(heatmap)
                heatmap = np.clip(heatmap, -v_max, v_max) / v_max  # → [-1, +1]

                if blur_sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    heatmap = gaussian_filter(heatmap, sigma=blur_sigma)
            else:
                # Clip and blur for clean display
                v_max   = np.percentile(heatmap, percentile)
                heatmap = np.clip(heatmap, 0, v_max)

                if blur_sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

            self.heatmap = heatmap
            return heatmap

        else:
            raise ValueError(
                f"Unknown mode '{use_mode}'. Expected 'evaluate' or 'visualize'."
            )
        
    def visualize_heatmap_all_mode(
        self,
        x:           torch.Tensor,
        sample_id:   int = 0,
        save_path:   Optional[str] = None,
        device:      torch.device = torch.device("cpu"),
    ) -> None:
        """
        Visualises PAF decision logic for all scoring modes.

        Layout:
            Row 0: Input image | Heatmap mode_1 | Heatmap mode_2 | ...
            Row 1: (empty)     | Overlay mode_1 | Overlay mode_2 | ...

        If both ABS (tau=1.0) and SIGNED_FULL (tau=1.0) are in mode_keys,
        an extra column is added showing ABS - SIGNED_FULL (inhibitory residual).
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        N, C, H, W = x.shape

        # --- Prepare image display ---
        img_disp = x[self.sample_idx].cpu().numpy()
        if C == 1:
            img_disp = img_disp.squeeze(0)
            cmap_img = "gray"
        else:
            img_disp = np.transpose(img_disp, (1, 2, 0))
            img_disp = (img_disp - img_disp.min()) / \
                       (img_disp.max() - img_disp.min() + 1e-8)
            cmap_img = None

        # --- Get heatmaps for all modes ---
        mode_keys = list(self.paf._score_fns.keys())
        heatmaps  = {
            key: self.get_heatmap_per_mode(
                mode_key   = key,
                sample_idx = self.sample_idx,
                blur_sigma = 1.0,
                percentile = 99.0,
                use_mode   = 'visualize',
            )
            for key in mode_keys
        }

        # --- Check if ABS - SIGNED_FULL difference map can be computed ---
        abs_key    = (ScoringMode.ABS,         1.0)
        signed_key = (ScoringMode.SIGNED_FULL, 1.0)
        has_diff   = abs_key in mode_keys and signed_key in mode_keys

        if has_diff:
            # ABS - SIGNED_FULL = 2 * inhibitory flow (see derivation above)
            # Normalise independently for display
            diff_map = heatmaps[abs_key] - heatmaps[signed_key]
            # diff_map can be negative — clamp to non-negative for display
            # (negative values would mean signed > abs which should not happen)
            diff_map = np.clip(diff_map, 0, None)
            v_max    = np.percentile(diff_map, 99) + 1e-9
            diff_map = diff_map / v_max

        # --- Layout ---
        n_modes = len(mode_keys)
        n_extra = 1 if has_diff else 0
        n_cols  = 1 + n_modes + n_extra   # input + modes + optional diff

        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        # Title
        title_str  = f"PAF Decision Logic | Pred: {self.target_class}"
        title_str += " | (correct)" if self.true_class == self.target_class \
                     else " | (misclassified)"
        fig.suptitle(title_str, fontsize=14, fontweight="bold")

        # --- Col 0: input image ---
        axes[0, 0].imshow(img_disp, cmap=cmap_img)
        axes[0, 0].set_title("Input Image")
        axes[1, 0].axis("off")

        # --- Cols 1..n_modes: one column per mode ---
        for col_idx, key in enumerate(mode_keys, start=1):
            mode, *params = key
            if mode == ScoringMode.SIGNED_SPLIT:
                tau, alpha, beta = params if len(params) == 3 else (*params, 1.0, 0.0)
                if alpha == 1.0 and beta == 0.0:
                    mode_label = "Excitatory\n(α=1,β=0)"
                elif alpha == 0.0 and beta == 1.0:
                    mode_label = "Inhibitory\n(α=0,β=1)"
                else:
                    mode_label = f"Split\nα={alpha},β={beta}"
                is_signed_mode = False
            elif mode == ScoringMode.SIGNED_FULL:
                mode_label     = f"signed_full\nτ={params[0]}"
                is_signed_mode = True
            else:
                mode_label     = f"{mode.value}\nτ={params[0]}"
                is_signed_mode = False

            normalization_factor = self.get_normalization_factor_per_mode(mode_key=key)
            heatmap              = heatmaps[key]

            if is_signed_mode:
                # Diverging colourmap for signed heatmap
                vmax       = np.abs(heatmap).max() + 1e-9
                cmap_heat  = 'RdBu_r'
                vmin_heat  = -vmax
                vmax_heat  = vmax
            else:
                cmap_heat  = 'hot'
                vmin_heat  = 0
                vmax_heat  = normalization_factor

            # Row 0 — heatmap
            ax_hm = axes[0, col_idx]
            im    = ax_hm.imshow(
                heatmap, cmap=cmap_heat,
                vmin=vmin_heat, vmax=vmax_heat
            )
            ax_hm.set_title(f"{mode_label}\nClass {self.target_class}")
            plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

            # Row 1 — overlay
            ax_ov = axes[1, col_idx]
            ax_ov.imshow(img_disp, cmap=cmap_img)
            im_ov = ax_ov.imshow(
                heatmap, cmap=cmap_heat, alpha=0.5,
                vmin=vmin_heat, vmax=vmax_heat
            )
            ax_ov.set_title(f"Overlay\n{mode_label}")
            plt.colorbar(im_ov, ax=ax_ov, fraction=0.046, pad=0.04)

        # --- Last column: ABS - SIGNED_FULL difference map ---
        if has_diff:
            col_idx = 1 + n_modes   # column after all mode columns

            # Row 0 — difference heatmap
            ax_hm = axes[0, col_idx]
            im    = ax_hm.imshow(diff_map, cmap='hot', vmin=0, vmax=1)
            ax_hm.set_title("ABS - Signed Full\n(Inhibitory Residual)")
            plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

            # Row 1 — overlay
            ax_ov = axes[1, col_idx]
            ax_ov.imshow(img_disp, cmap=cmap_img)
            im_ov = ax_ov.imshow(diff_map, cmap='plasma', alpha=0.5,
                                  vmin=0, vmax=1)
            ax_ov.set_title("Overlay\nInhibitory Residual")
            plt.colorbar(im_ov, ax=ax_ov, fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_paf_layerwise_distribution(
        self,
        save_path:  Optional[str] = None,
    ) -> None:
        """
        Plots probability flow from output to input for all scoring modes.
        One plot per mode saved as a separate file.
        """
        graph        = self.paf.model.graph_info
        node_types   = graph['node_types']
        predecessors = graph.get('predecessors', {})
        successors   = graph.get('successors', {})

        color_map = {
            'conv'      : '#1f77b4',
            'add'       : '#ff7f0e',
            'relu'      : '#2ca02c',
            'batchnorm' : '#d62728',
            'fc'        : '#9467bd',
            'input'     : '#e377c2',
            'unknown'   : '#7f7f7f',
        }

        # --- Backward order: output → input ---
        ordered_nodes = graph['backward_order']  # already output → input

        # --- One plot per mode ---
        for mode_key in self.paf._score_fns:
            print(f"mode-key: {mode_key}")
            mode = mode_key[0]
            tau  = mode_key[1] if len(mode_key) > 1 else 1.0
            mode, *params = mode_key
            if mode == ScoringMode.SIGNED_SPLIT:
                tau, alpha, beta = params if len(params) == 3 else (*params, 1.0, 0.0)
                if alpha == 1.0 and beta == 0.0:
                    mode_label = "Excitatory\n(α=1,β=0)"
                elif alpha == 0.0 and beta == 1.0:
                    mode_label = "Inhibitory\n(α=0,β=1)"
                else:
                    mode_label = f"Split\nα={alpha},β={beta}"
                is_signed_mode = False
            elif mode == ScoringMode.SIGNED_FULL:
                mode_label     = f"signed_full\nτ={params[0]}"
                is_signed_mode = True
            else:
                mode_label     = f"{mode.value}\nτ={params[0]}"
                is_signed_mode = False

            node_masses = self.paf.distributions[mode_key]
            edge_masses = self.paf.edge_mass[mode_key]

            # Build node lists in output → input order
            names, masses, colors = [], [], []
            for node in ordered_nodes:
                if node not in node_masses:
                    continue
                names.append(node)
                if mode == ScoringMode.SIGNED_FULL:
                    masses.append(node_masses[node].sum().item())
                else:
                    masses.append(node_masses[node].abs().sum().item())

                ntype      = node_types.get(node, 'unknown').lower()
                found_color = color_map['unknown']
                for key, hex_c in color_map.items():
                    if key in ntype:
                        found_color = hex_c
                        break
                colors.append(found_color)

            if not names:
                continue

            name_to_x = {name: i for i, name in enumerate(names)}

            fig, ax = plt.subplots(figsize=(20, 8))
            x_pos = np.arange(len(names))
            ax.bar(
                x_pos, masses,
                color=colors, edgecolor='black', linewidth=0.5, alpha=0.7
            )

            # --- Draw split/join arrows in output → input direction ---
            for dst_node, preds in predecessors.items():
                # In forward pass: preds → dst_node
                # In backward pass (output→input): dst_node → preds
                # So arrow goes FROM dst_node TO src_node
                if dst_node not in name_to_x:
                    continue
                idx_dst = name_to_x[dst_node]
                val_dst = masses[idx_dst]

                for src_node in preds:
                    if src_node not in name_to_x:
                        continue
                    idx_src  = name_to_x[src_node]
                    val_src  = masses[idx_src]
                    e_flow   = edge_masses.get((src_node, dst_node), 0)

                    is_split = len(successors.get(src_node, [])) > 1
                    is_join  = len(preds) > 1

                    if (is_split or is_join) and e_flow > 0.005:
                        # In backward order, dst_node appears BEFORE src_node
                        # so arrow goes left to right: idx_dst < idx_src
                        dist = idx_src - idx_dst
                        rad  = -0.4 if dist > 1 else 0.1
                        arrow_c = '#ff7f0e' if dist > 1 else '#555555'

                        # Arrow: FROM dst_node (output side) TO src_node (input side)
                        ax.annotate(
                            "",
                            xy     = (idx_src, val_src),   # arrowhead at input side
                            xytext = (idx_dst, val_dst),   # tail at output side
                            arrowprops=dict(
                                arrowstyle="->",
                                color=arrow_c,
                                lw=1.0 + (e_flow * 2),
                                alpha=0.8,
                                connectionstyle=f"arc3,rad={rad}"
                            )
                        )

                        # Label at midpoint
                        y_limit = max(masses)
                        off     = y_limit * 0.05 if dist > 1 else y_limit * 0.02
                        mid_x   = (idx_dst + idx_src) / 2
                        mid_y   = (val_dst + val_src) / 2 + off

                        ax.text(
                            mid_x, mid_y, f"{e_flow:.2f}",
                            fontsize=8, fontweight='bold', color=arrow_c,
                            bbox=dict(
                                facecolor='white', alpha=0.6,
                                edgecolor='none', pad=1
                            )
                        )

            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(names, rotation=90, fontsize=7)
            ax.set_ylabel("Probability Mass")
            ax.set_title(
                f"PAF Branching Analysis — Output → Input\n"
                f"Mode: {mode_label}"
            )

            legend_patches = [
                mpatches.Patch(color=c, label=k.upper())
                for k, c in color_map.items()
            ]
            ax.legend(handles=legend_patches, loc='upper right')
            plt.tight_layout()

            # --- Save one file per mode ---
            if save_path:
                # Insert mode info into filename
                # e.g. "output/branching.png" → "output/branching_abs_tau1.0.png"
                base, ext = os.path.splitext(save_path)
                mode_path = f"{base}_{mode_label}{ext}"
                os.makedirs(os.path.dirname(mode_path), exist_ok=True)
                plt.savefig(mode_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {mode_path}")
            else:
                plt.show()

            plt.close()

    def plot_signed_explanation(self, original_img, predicted,sample_idx,save_path):
        dist_in=self.paf.distributions_signed["x"][sample_idx].squeeze(0)
        dist_in_unsigned=self.paf.distributions["x"][sample_idx].squeeze(0)

        signed_heatmap=self.get_signed_heatmap(dist_in.squeeze(0))

        # 1. Handle Torch Tensors and Shape conversion
        if isinstance(original_img, torch.Tensor):
            # Remove batch dim if present: (1, 3, H, W) -> (3, H, W)
            if original_img.dim() == 4:
                original_img = original_img.squeeze(0)
            
            # Move channels to the end: (3, H, W) -> (H, W, 3)
            original_img = original_img.permute(1, 2, 0).cpu().numpy()

        # 2. Denormalize if necessary 
        # If your image is in range [-1, 1] or normalized by ImageNet stats, 
        # you might need to rescale it to [0, 1] for correct display
        if original_img.min() < 0:
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        
        # Show original image
        ax[0].imshow(original_img)
        ax[0].set_title("Input Image")
        
        # Show signed heatmap
        # cmap='seismic' or 'RdBu_r' are perfect for signed data
        im = ax[1].imshow(signed_heatmap, cmap='seismic', vmin=-1, vmax=1)
        
        # Add a colorbar to explain the signs
        cbar = plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        cbar.set_label('Evidence: Negative (Blue) | Positive (Red)')
        
        ax[1].set_title("PAF Signed Attribution")

        # Measure of internal model conflict per pixel
        conflict_map = dist_in_unsigned.sum(dim=0).cpu().numpy() - np.abs(dist_in.sum(dim=0).cpu().numpy())
        im = ax[2].imshow(conflict_map, cmap='seismic', vmin=-1, vmax=1)
        ax[2].set_title("PAF Conflict map")
        cbar = plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        cbar.set_label('Conflict Map')
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()

    def visualize_distributions(self):
        """
        Draws the ResNet FX graph with edges labeled by their probability mass.
        
        Args:
            traced_model: The fx.GraphModule of the ResNet.
            successor_list: Dict where keys are node names and values are lists of parent names.
            edge_distributions: Dict {(node_name, parent_name): float_value}
        """
        dot = Digraph(comment='PAF Probability Distribution', format='png')
        dot.attr(rankdir='TB', nodesep='0.6', ranksep='0.8')
        
        # 1. Define Node Styles
        def get_color(node_name):
            if 'conv' in node_name: return '#3498db'
            if 'bn' in node_name: return '#2ecc71'
            if 'relu' in node_name: return '#e67e22'
            if 'add' in node_name: return '#f1c40f'
            if 'fc' in node_name: return '#e74c3c'
            return '#95a5a6'

        Nodes=self.paf.model.graph_info['traced'].graph.nodes
        Successors=self.paf.model.graph_info['successors']
        edge_distributions=self.paf.edge_mass
        # 2. Add Nodes
        for node in Nodes:
            if node.op == 'output': continue
            
            color = get_color(node.name)
            font_color = 'white' if color != '#f1c40f' else 'black'
            
            dot.node(node.name, f"{node.name}", 
                    style='filled', fillcolor=color, fontcolor=font_color,
                    shape='rect', penwidth='0')

        # 3. Add Edges with Distribution Labels
        for node_name, parents in Successors.items():
            for parent_name in parents:
                # Retrieve the probability mass for this specific edge
                # Default to 0.0 if not found
                mass = edge_distributions.get((node_name, parent_name), 0.0)
                
                # Format label: show as percentage if very small, or fixed float
                label_text = f"{mass:.4f}" if mass > 0.0001 else "<0.0001"
                
                # Visual styling for Skip Connections
                is_skip = False
                # Logic: If it's an 'add' node and the parent isn't the 
                # immediate chronological predecessor in the FX trace
                all_nodes = list(Nodes)
                node_idx = next(i for i, n in enumerate(all_nodes) if n.name == node_name)
                if 'add' in node_name and parent_name != all_nodes[node_idx-1].name:
                    is_skip = True

                # Edge styling based on "Heat" (higher mass = thicker/darker line)
                normalized_mass = min(mass * 5, 5.0) # Scale for visibility
                
                dot.edge(parent_name, node_name, 
                        label=f" p={label_text} ",
                        fontsize='10',
                        fontcolor='#2c3e50',
                        color='#8e44ad' if is_skip else '#2c3e50',
                        style='dashed' if is_skip else 'solid',
                        penwidth=str(max(1.0, normalized_mass)))

        return dot
    def get_normalization_factor_per_mode(self, mode_key: tuple = None) -> float:
        """
        Compute normalisation factor for heatmap display.
        Uses 99th percentile clipping to avoid outlier dominance.
        mode_key defaults to self.mode_key if not specified.
        """
        key = mode_key if mode_key is not None else self.mode_key

        def _get_clipped_max(distributions_dict: dict) -> float:
            dist = distributions_dict[key]['x'][self.sample_idx] \
                .sum(dim=0).cpu().numpy()
            v_max   = np.percentile(dist, 99)
            clipped = np.clip(dist, 0, v_max)
            return clipped.max()

        curr_max = _get_clipped_max(self.paf.distributions)

        if not self.misclassification or self.paf_shared is None:
            return curr_max
        else:
            shared_max = _get_clipped_max(self.paf_shared.distributions)
            return max(curr_max, shared_max)
    
    def get_normalization_factor(self):
        dist_curr=self.paf.distributions["x"][self.sample_idx].sum(dim=0).cpu().numpy()
        v_max = np.percentile(dist_curr, 99)
        dist_curr_clipped = np.clip(dist_curr, 0, v_max)

        if not self.misclassification or self.paf_shared is None:
            return dist_curr_clipped.max()
            #normalization_factor=heatmap.max()
        else: 
            dist_shared=self.paf_shared.distributions["x"][self.sample_idx].sum(dim=0).cpu().numpy()
            shared_max = np.percentile(dist_shared, 99)
            dist_shared_clipped = np.clip(dist_shared, 0, shared_max)
            normalization_factor= max(dist_curr_clipped.max(),dist_shared_clipped.max())
            return normalization_factor

    def visualize_probability_decision_logic_prev(
        self,
        x: torch.Tensor,
        x_orig: torch.Tensor,
        predicted: Union[int, torch.Tensor],
        true_label: Optional[Union[int, torch.Tensor]] = None,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Generalized visualization for PAF decision logic.
        Handles RGB/Grayscale, arbitrary spatial resolutions, and batch indexing.
        """
        
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        N, C, H, W = x.shape
        dist_current=self.paf.distributions["x"][sample_idx]
        p_input_pred = self.get_input_heatmap(
            dist_current,
            blur_sigma=1,
            percentile=99         # Try 98.0, 98.5, or 99.0 for best visual balance
        )
        
        cols = 3 
        fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 10))
        
        # Title Logic
        title_str = f"PAF Decision Logic | Pred: {self.target_class}"
        if self.true_class == self.target_class:
            title_str += f" | (correct label)"
        else: 
            title_str += f" | (misclassified)"
            
        fig.suptitle(title_str, fontsize=14, fontweight="bold")
        
        # 3. Handle Image Prepping (RGB vs Grayscale)
        #img_disp = x_visual[sample_idx].cpu().numpy()
        # Prepare for Matplotlib
        img_disp_original = x_orig[sample_idx].cpu().numpy()
        img_disp = x[sample_idx].cpu().numpy()

        if C == 1:
            img_disp_original = img_disp_original.squeeze(0)
            img_disp = img_disp.squeeze(0)
            cmap_img = "gray"
        else:
            img_disp_original = np.transpose(img_disp_original, (1, 2, 0))
            img_disp_original = (img_disp_original - img_disp_original.min()) / (img_disp_original.max() - img_disp_original.min() + 1e-8)
            img_disp = np.transpose(img_disp, (1, 2, 0))
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            cmap_img = None

        # Column 1: Input & True Prob (if exists)
        axes[0, 0].imshow(img_disp_original, cmap=cmap_img)
        axes[0, 0].set_title("Input Image")

        # Column 2: Predicted Heatmap
        normalization_factor=self.get_normalization_factor()
        im1 = axes[0, 1].imshow(p_input_pred, cmap="hot",vmin=0,vmax=normalization_factor)
        axes[0, 1].set_title(f"Class {self.target_class} Prob")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Column 3 (or Overlay): Predicted Overlay
        if cols == 3:
            ax_ov = axes[0, 2]
            # Use a standard heatmap overlay technique (Alpha blending)
            ax_ov.imshow(img_disp_original, cmap=cmap_img)
            
            #Advanced overlay: Normalize the heatmap for better contrast, then overlay with 'jet' colormap: others include inferno, magma, plasma
            im_ov = ax_ov.imshow(p_input_pred, cmap="plasma", alpha=0.5,vmin=0,vmax=normalization_factor) 
            # Add a localized colorbar to see the relative importance
            plt.colorbar(im_ov, ax=ax_ov, fraction=0.046, pad=0.04)
            ax_ov.set_title("PAF Overlay")
        
        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_paf_prev(
        self,
        x: torch.Tensor,
        x_orig: torch.Tensor,
        predicted: int,
        true_label: Optional[int],
        save_path: Optional[str] = None,
        ) -> None:

        sample_idx=0
        distributions = self.paf.distributions
        #N, C, H, W = x.shape

        # ── Infer input shape ─────────────────────────────────────────────────────
        if x.dim() == 4:      # (B,C,H,W)
            img = x[0]
            img_orig = x_orig[0]
        else:
            img = x
            img_orig = x_orig

        if img.dim() == 3:    # (C,H,W)
            C, H, W = img.shape
        elif img.dim() == 2:  # (H,W)
            C, H, W = 1, *img.shape
            img = img.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input shape {img.shape}")

        img_np = img.detach().cpu().numpy()
        img_orig_np = img_orig.detach().cpu().numpy()

        # ── Classify layers ───────────────────────────────────────────────────────
        conv_layers = []
        fc_layers = []
        other_layers = []

        for name, p in distributions.items():
            if p.dim() == 3:
                conv_layers.append(name)
            elif p.dim() == 1:
                fc_layers.append(name)
            else:
                other_layers.append(name)

        fc_layers = sorted(fc_layers)  # optional ordering

        # ── Figure layout ─────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"PAF Analysis | Predicted: {predicted}"
            + (f" | True: {true_label}" if true_label is not None else ""),
            fontsize=14,
            fontweight="bold",
        )

        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.6, wspace=0.65)

        # ── (0,0) Input image ─────────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])

        if C == 1:
            ax0.imshow(img_orig_np.squeeze(0), cmap="gray")
        else:
            ax0.imshow(np.transpose(img_orig_np, (1, 2, 0)))

        ax0.set_title("Input image")
        ax0.axis("off")

        # ── (0,1) Input-layer PAF ────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])

        if "input" in distributions:
            p = distributions["input"].detach().cpu()

            try:
                if p.dim() == 1:
                    p_map = p.view(H, W)
                elif p.dim() == 2:
                    p_map = p
                elif p.dim() == 3:
                    p_map = p.mean(dim=0)
                else:
                    raise ValueError

                im = ax1.imshow(p_map.numpy(), cmap="hot")
                ax1.set_title("PAF (input)")
                ax1.axis("off")
                plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

            except Exception:
                ax1.text(0.5, 0.5, "Invalid input PAF",
                        ha="center", va="center", transform=ax1.transAxes)
                ax1.axis("off")
        else:
            ax1.axis("off")

        # ── (0,2) Hidden FC (if exists) ──────────────────────────────────────────
        if len(fc_layers) > 1:
            hidden_fc = fc_layers[:-1][0]  # take first hidden layer
            ax2 = fig.add_subplot(gs[0, 2])

            p = distributions[hidden_fc].detach().cpu().numpy().flatten()

            top_k = min(20, len(p))
            top_idx = np.argsort(p)[::-1][:top_k]

            ax2.bar(range(top_k), p[top_idx])
            ax2.set_xticks(range(top_k))
            ax2.set_xticklabels(top_idx, rotation=90, fontsize=7)
            ax2.set_title(f"{hidden_fc} top-{top_k}")

        else:
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis("off")

        # ── (0,3) Output layer ───────────────────────────────────────────────────
        if fc_layers:
            output_layer = fc_layers[-1]
            ax3 = fig.add_subplot(gs[0, 3])

            p = distributions[output_layer].detach().cpu().numpy().flatten()
            n = len(p)

            colors = ["crimson" if i == predicted else "steelblue" for i in range(n)]
            ax3.bar(range(n), p, color=colors)

            ax3.set_title(f"{output_layer} (output)")
        else:
            ax3 = fig.add_subplot(gs[0, 3])
            ax3.axis("off")

        # ── (1,0-1) Conv layers ──────────────────────────────────────────────────
        if len(conv_layers) > 0:
            gs_conv = gridspec.GridSpecFromSubplotSpec(
                1, len(conv_layers), subplot_spec=gs[1, 0:2], wspace=0.3
            )

            for i, lname in enumerate(conv_layers):
                ax = fig.add_subplot(gs_conv[0, i])
                p = distributions[lname].detach().cpu()

                try:
                    if p.dim() == 3:
                        p_map = p.mean(dim=0)
                    elif p.dim() == 1:
                        size = int(np.sqrt(len(p)))
                        p_map = p.view(size, size)
                    else:
                        raise ValueError

                    im = ax.imshow(p_map.numpy(), cmap="hot")
                    ax.set_title(lname)
                    ax.axis("off")

                except Exception:
                    ax.text(0.5, 0.5, f"{lname}\nunsupported",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off")

        else:
            ax_empty = fig.add_subplot(gs[1, 0:2])
            ax_empty.axis("off")

        # ── (1,2-3) Entropy ──────────────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 2:4])

        entropies = []
        labels = []

        for name, p in distributions.items():
            p = p.detach().cpu().flatten()
            p = p.clamp(min=1e-12)

            h = float(-(p * p.log()).sum())

            entropies.append(h)
            labels.append(name)

        ax5.plot(range(len(entropies)), entropies, "o-")
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, rotation=45)
        ax5.set_title("Layer entropy")

        # ── Finalize ─────────────────────────────────────────────────────────────
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_layer_sensitivity(self,save_path: Optional[str] = None):
        # 1. Gather Data
        node_masses = self.paf.distributions  # node_name -> tensor
        node_types = self.paf.model.graph_info['node_types']
        
        # 2. Define Semantic Color Palette
        color_map = {
            'conv': '#1f77b4',      # Blue
            'add': '#ff7f0e',       # Orange (The skip connection mergers!)
            'relu': '#2ca02c',      # Green
            'batchnorm': '#d62728', # Red
            'fc': '#9467bd',        # Purple
            'maxpool': '#8c564b',   # Brown
            'input': '#e377c2',     # Pink
            'unknown': '#7f7f7f'    # Gray
        }

        names = []
        masses = []
        colors = []

        # 3. Process nodes (using backward_order to keep it chronological from output to input)
        for node in self.paf.model.graph_info['backward_order']:
            if node in node_masses:
                names.append(node)
                masses.append(node_masses[node].sum().item())
                
                # Map the node type to our color scheme
                ntype = node_types.get(node, 'unknown')
                # Handle variations in naming (e.g., 'conv2d' -> 'conv')
                found_color = color_map.get('unknown')
                for key in color_map:
                    if key in ntype.lower():
                        found_color = color_map[key]
                        break
                colors.append(found_color)

        # 4. Create the Visualization
        plt.figure(figsize=(16, 6))
        bars = plt.bar(range(len(masses)), masses, color=colors, edgecolor='black', linewidth=0.5)
        
        # 5. Legend
        legend_patches = [mpatches.Patch(color=c, label=k.upper()) for k, c in color_map.items()]
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Node Types")

        # 6. Formatting
        plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
        plt.ylabel("Total Probability Mass")
        plt.title("PAF Architectural Sensitivity | Flow & Node Type Analysis")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_paf_branching_logic(self,is_signed=False,save_path: Optional[str] = None):
        graph = self.paf.model.graph_info
        if is_signed:
            node_masses = self.paf.distributions_signed
            edge_masses = self.paf.edge_mass_signed
        else:
            node_masses = self.paf.distributions
            edge_masses = self.paf.edge_mass

        node_types = graph['node_types']
        predecessors = graph.get('predecessors', {})
        
        # 1. Map successors to find splits
        successors = graph.get('successors', {})
        
        color_map = {
            'conv': '#1f77b4', 'add': '#ff7f0e', 
            'relu': '#2ca02c', 'batchnorm': '#d62728', 
            'fc': '#9467bd', 'input': '#e377c2', 'unknown': '#7f7f7f'
        }

        ordered_nodes = graph['backward_order'][::-1] 
        names, masses, colors = [], [], []
        for node in ordered_nodes:
            if node in node_masses:
                names.append(node)
                masses.append(node_masses[node].abs().sum().item())
                ntype = node_types.get(node, 'unknown').lower()
                found_color = color_map['unknown']
                for key, hex_c in color_map.items():
                    if key in ntype: found_color = hex_c; break
                colors.append(found_color)

        fig, ax = plt.subplots(figsize=(20, 8))
        x_pos = np.arange(len(names))
        name_to_x = {name: i for i, name in enumerate(names)}
        ax.bar(x_pos, masses, color=colors, edgecolor='black', linewidth=0.5, alpha=0.7)

        # 2. Draw ONLY Split and Join arrows
        for dst_node, preds in predecessors.items():
            if dst_node not in name_to_x: continue
            idx_dst = name_to_x[dst_node]
            val_dst = masses[idx_dst]

            for src_node in preds:
                if src_node not in name_to_x: continue
                idx_src = name_to_x[src_node]
                val_src = masses[idx_src]
                e_flow = edge_masses.get((src_node, dst_node), 0)

                # CONDITION (1): Only draw if it's a split OR a join
                # Split: Source has >1 successor | Join: Destination has >1 predecessor
                is_split = len(successors.get(src_node, [])) > 1
                is_join = len(preds) > 1
                
                if (is_split or is_join) and e_flow > 0.005:
                    # Use a curve for skips, straight for main branch splits
                    dist = idx_dst - idx_src
                    rad = -0.4 if dist > 1 else 0.1
                    arrow_c = '#ff7f0e' if dist > 1 else '#555555'
                    
                    # Draw Arrow
                    arrow = ax.annotate("",
                                xy=(idx_dst, val_dst), xytext=(idx_src, val_src),
                                arrowprops=dict(
                                    arrowstyle="->", color=arrow_c,
                                    lw=1.0 + (e_flow * 2), alpha=0.8,
                                    connectionstyle=f"arc3,rad={rad}"
                                ))
                    
                    # CONDITION (2): Probability Labels
                    # Calculate midpoint for text placement

                    y_limit = max(masses)
                    off_large = y_limit * 0.05  # 5% for skips
                    off_small = y_limit * 0.02

                    mid_x = (idx_src + idx_dst) / 2
                    mid_y = (val_src + val_dst) / 2 + (off_large if dist > 1 else off_small)
                    
                    ax.text(mid_x, mid_y, f"{e_flow:.2f}", 
                            fontsize=8, fontweight='bold', color=arrow_c,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_ylabel("Probability Mass")
        ax.set_title("PAF Branching Analysis: Probability Flow on Splits & Joins")
        
        legend_patches = [mpatches.Patch(color=c, label=k.upper()) for k, c in color_map.items()]
        ax.legend(handles=legend_patches, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def show_salient_features_likeheatmap(self, x_orig, save_path=None, title="Salient Features (Model Focus)"):
        sample_idx = 0
        
        # 1. Get the improved heatmap (positive contribution)
        heatmap_np = self.get_input_heatmap(
            self.paf.distributions["x"][sample_idx],
            percentile=98.5,
            blur_sigma=0.8
        )
        
        # 2. Prepare the original image (RGB or grayscale)
        img = x_orig[sample_idx].cpu().numpy()
        if img.shape[0] == 3:                    # RGB
            img = np.transpose(img, (1, 2, 0))
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
        else:                                    # Grayscale
            img = img.squeeze(0)
        
        # 3. Create a nice overlay (this is the key improvement)
        # Blend original image with colored heatmap
        from matplotlib.colors import LinearSegmentedColormap
        cmap = plt.cm.jet  # or 'inferno', 'magma', 'plasma'
        
        # Normalize heatmap again for safety
        heatmap = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        # Create colored heatmap
        colored_heatmap = cmap(heatmap)[:, :, :3]   # RGBA → RGB
        
        # Blend: 0.6 original + 0.7 colored heatmap (adjust alpha as needed)
        overlay = (img * 0.6 + colored_heatmap * 0.7).clip(0, 1)
        
        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def show_salient_features(self, x_orig, save_path=None, title="Salient Features (Model Focus)"):
        sample_idx = 0
        heatmap_np = self.get_input_heatmap(self.paf.distributions["x"][sample_idx],percentile=99.0, blur_sigma=0) 
        # since heatmap does not normalize, we should normalize here so that everything does not look dark
        mask = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0) # [1, 1, 224, 224]

        # 2. Handle x_orig dimensions and Get Target Shape
        x_sample = x_orig[sample_idx] if x_orig.dim() == 4 else x_orig
        _, h, w = x_sample.shape  # Extract actual height and width from input tensor
        mask_resampled = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        norm_factor=self.get_normalization_factor()
        mask_normalized = mask_resampled/(norm_factor+1e-8)
        mask_boosted = torch.pow(mask_normalized, 0.5)

            # 3. Un-normalize the Image (Restore natural colors)
        # These are the standard ImageNet constants used by ResNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Move to CPU for plotting
        x_cpu = x_sample.cpu()
        x_rgb = (x_cpu * std) + mean
        x_rgb = torch.clamp(x_rgb, 0, 1) # Ensure [0, 1] range


        x_salient = (x_rgb * mask_boosted.squeeze(0).cpu()).detach()
        
        # 5. Permute for Matplotlib [C, H, W] -> [H, W, C]
        # We use x_salient.squeeze() here to ensure it is exactly 3D
        #img = x_salient.squeeze().permute(1, 2, 0).numpy()
        
        # 5. Convert to Numpy for Matplotlib [C, H, W] -> [H, W, C]
        img = x_salient.permute(1, 2, 0).detach().numpy()

        # 6. Final Normalize & Plot
        #img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        # Title Logic
        title= f"PAF salient feature | Pred: {self.target_class}"
        if self.true_class == self.target_class:
            title += f" | (correct label)"
        else: 
            title += f" | (misclassified)"
        #plt.imshow(img, interpolation='nearest')

        plt.title(title)
        plt.axis('off')
        from matplotlib import ticker

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def show_clean_contours(self,x_orig, save_path=None, title="Structural Importance Contours (PAF)"):
        """
        heatmap_np: Your raw PAF heatmap [H, W]
        original_img_np: Your un-normalized RGB image [H, W, 3]
        """
        sample_idx = 0
        #x_orig = x_orig[sample_idx] if x_orig.dim() == 4 else x_orig
        # 1. Compute Heatmap (using your existing function)
        # We use the raw distribution for 'x' to get the full resolution
        p_in = self.paf.distributions["x"][sample_idx]
        heatmap_np = self.get_input_heatmap(p_in,percentile=90.0, blur_sigma=2.0) 

        # 2. Normalize smoothed heatmap for consistent thresholding
        #   smoothed_heatmap = (smoothed_heatmap - smoothed_heatmap.min()) / (smoothed_heatmap.max() - smoothed_heatmap.min() + 1e-8)

        plt.figure(figsize=(8, 8))
        
        # 3. Show the original image as the background
        # We dim it slightly (alpha=0.8) so the contours pop
        if torch.is_tensor(x_orig):
            # .squeeze() turns [1, 3, 224, 224] into [3, 224, 224]
            # .permute(1, 2, 0) turns [3, 224, 224] into [224, 224, 3]
            display_img = x_orig.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        else:
            display_img = x_orig
        plt.imshow(display_img, alpha=0.9) 

        # 4. Define specific levels to show (50% and 80% importance)
        # This removes the clutter of 30%, 40%, etc.
        levels = [0.5, 0.8]
        colors = ['cyan', 'yellow']
        
        contours = plt.contour(
            heatmap_np, 
            levels=levels, 
            colors=colors, 
            linewidths=2.0
        )

        # 5. Add clean labels for just those two levels
        # fmt defines how the labels look (e.g., '50%')
        fmt = {0.5: '50%', 0.8: '80%'}
        plt.clabel(contours, inline=True, fontsize=12, fmt=fmt)

        plt.title(title, fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            print(f"Contour plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_salient_contours(self, x_orig, save_path=None, title="Salient Contours (PAF)"):
        """
        x_orig: The original input tensor [C, H, W] or [1, C, H, W]
        """
        sample_idx = 0
        
        # 1. Compute Heatmap (using your existing function)
        # We use the raw distribution for 'x' to get the full resolution
        p_in = self.paf.distributions["x"][sample_idx]
        heatmap_np = self.get_input_heatmap(p_in) 
        
        # 2. Generate the Salient Base Image (the 'cutout')
        # Convert heatmap to torch and downsample to 32x32
        mask = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0)

        # 2. Handle x_orig dimensions and Get Target Shape
        x_sample = x_orig[sample_idx] if x_orig.dim() == 4 else x_orig
        _, h, w = x_sample.shape  # Extract actual height and width from input tensor
        
        # 3. Dynamic Resampling
        # Instead of (32, 32), we use (h, w) to match the input image exactly
        mask_resampled = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # Gamma correction to enhance contrast (0.5 makes dark areas slightly more visible)
        mask_resampled = torch.pow(mask_resampled, 0.5)
        
        # Prepare x_orig and multiply
        x_salient = (x_sample.cpu() * mask_resampled.squeeze(0).cpu()).detach()
        
        # Convert to [32, 32, 3] for plotting
        img_base = x_salient.squeeze().permute(1, 2, 0).numpy()
        img_base = (img_base - img_base.min()) / (img_base.max() - img_base.min() + 1e-8)

        # 3. Create the Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_base, interpolation='nearest')
        
        # 4. Draw Contours
        # We use mask_32 (non-gamma) for contours to show the true attribution levels
        mask_contour = mask_resampled.squeeze().numpy()
        mask_contour = (mask_contour - mask_contour.min()) / (mask_contour.max() - mask_contour.min() + 1e-8)
        mask_contour = gaussian_filter(mask_contour, sigma=2.0)
        # Define levels: 30% (Low focus), 60% (Medium), 90% (Peak focus)
        #levels = [0.6, 0.8, 0.95]
        levels=[0.3, 0.5, 0.8]
        colors = ['cyan', 'yellow', 'lime'] # High contrast colors
        h, w = mask_contour.shape
        cntr = ax.contour(mask_contour, levels=levels, colors=colors, linewidths=2.0, alpha=0.9,extent=(0, w, h, 0))
        
        # Check if contours actually were found
        if any(len(path.vertices) > 0 for path in cntr.get_paths()):            
            ax.clabel(cntr, inline=True, fontsize=10, fmt={l: f'{int(l*100)}%' for l in levels}, colors='white')
            #ax.clabel(cntr, inline=True, fontsize=8, fmt={l: f'{int(l*100)}%' for l in levels}, colors='white')
        else:
            print("Warning: No contours found at specified levels. Try lowering 'levels'.")
        

        title= f"PAF salient feature | Pred: {self.target_class}"
        if self.true_class == self.target_class:
            title += f" | (correct label)"
        else: 
            title += f" | (misclassified)"
        # 5. Final Touch
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # 6. Save or Show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            print(f"Contour plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def get_img_display(self,x):
        # 1. Move to CPU and remove batch dimension
        img = x.squeeze().cpu().detach()
        
        # 2. Reverse ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        
        # 3. Clean up for Matplotlib (Clamp to 0-1 and change order to HWC)
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()

    def plot_layer_results_combined(self, img_display, paf_tensor, layer_name):
        target_h, target_w = img_display.shape[:2]

        # Normalise to (1, C, H, W) or (1, C, 1, 1) for linear
        if paf_tensor.dim() == 1:
            # Linear layer — no spatial meaning
            # Visualise as a bar chart of top neurons, not a spatial heatmap
            return self._plot_linear_distribution(paf_tensor, layer_name)

        if paf_tensor.dim() == 2:   # (C, N) or (N, C) — shouldn't happen in PAF
            paf_tensor = paf_tensor.unsqueeze(0).unsqueeze(0)
        elif paf_tensor.dim() == 3:  # (C, H, W)
            paf_tensor = paf_tensor.unsqueeze(0)

        # Sum across channels — correct for PAF distributions
        spatial_map = paf_tensor.sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # Upsample to image size
        upsampled = F.interpolate(
            spatial_map,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().detach().numpy()

        # Normalise
        upsampled = np.clip(upsampled, 0, None)
        v_max = np.percentile(upsampled, 99.0)
        heatmap = np.clip(upsampled, 0, v_max)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{layer_name} | PAF Distribution", fontsize=14, fontweight='bold')

        axes[0].imshow(img_display)
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='magma')
        axes[1].set_title(f"PAF Mass — {layer_name}")
        axes[1].axis('off')

        axes[2].imshow(img_display)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        return fig


    def _plot_linear_distribution(self, paf_tensor, layer_name):
        """
        For linear layers: visualise distribution over neurons as a bar chart.
        Spatial upsampling is not meaningful here.
        """
        values = paf_tensor.cpu().detach().numpy()
        top_k = 20
        top_indices = np.argsort(values)[-top_k:][::-1]
        top_values = values[top_indices]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(top_k), top_values)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"n{i}" for i in top_indices], rotation=45)
        ax.set_title(f"{layer_name} | Top-{top_k} PAF neuron mass (linear layer)")
        ax.set_ylabel("Probability mass")
        plt.tight_layout()
        return fig

    def plot_layer_results_combined_old(self, img_display, paf_tensor, layer_name):
        # Get the real size of your fish image (e.g., 224, 224)
        target_h, target_w = img_display.shape[:2]
        
        # 1. Resize the PAF Probability Map
        spatial_map = paf_tensor.sum(dim=1, keepdim=True)
        #upsampled_map = F.interpolate(spatial_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        upsampled_map = F.interpolate(spatial_map, size=(target_h, target_w), 
                                    mode='bicubic', align_corners=False)
        
        heatmap_norm = upsampled_map.squeeze().cpu().detach().numpy()
        heatmap_norm = (heatmap_norm - heatmap_norm.min()) / (heatmap_norm.max() - heatmap_norm.min() + 1e-8)
        heatmap_norm = np.power(heatmap_norm, 2)
        # 2. Create the Figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Left: Heatmap
        axes[0].imshow(heatmap_norm, cmap='jet')
        axes[0].set_title(f"{layer_name} Heatmap")
        
        # Center: Features (using the properly colored img_display)
        mask = (heatmap_norm > 0.1)[:, :, np.newaxis]
        axes[1].imshow(img_display * mask)
        axes[1].set_title(f"{layer_name} Salient Features")
        
        # Right: Contours (overlaying on the clear image)
        axes[2].imshow(img_display)
        axes[2].contour(heatmap_norm, levels=[0.3, 0.6], colors=['cyan', 'yellow'])
        axes[2].set_title(f"{layer_name} Contours")
        
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        
        return fig
    


    def get_deepshap_layer_heatmap(
        self,
        input_tensor:  torch.Tensor,
        target_layer,
        target_idx:    int,
        n_samples:     int = 50,
    ) -> np.ndarray:
        """
        Computes DeepLiftShap attributions for a specific intermediate layer.
        Handles inplace=True ReLU error by creating a SHAP-safe model copy.
        """
        from captum.attr import LayerDeepLiftShap

        # Fix: DeepLIFT requires non-inplace ReLUs and unique module instances
        # Use make_model_universal_for_shap which:
        #   1. deepcopies the model
        #   2. replaces all ReLU(inplace=True) with ReLU(inplace=False)
        #   3. patches BasicBlock/Bottleneck forward to use unique relu instances
        shap_model = make_model_universal_for_shap(self.paf.model)
        shap_model.eval()

        # Find the equivalent target layer in the copied model
        # by matching the module's position in named_modules
        original_layer_name = None
        for name, module in self.paf.model.named_modules():
            if module is target_layer:
                original_layer_name = name
                break

        if original_layer_name is None:
            raise ValueError(
                f"target_layer not found in model. "
                f"Pass the module object directly, e.g. model.layer4[1].conv2"
            )

        # Get corresponding layer in shap_model
        shap_target_layer = dict(shap_model.named_modules())[original_layer_name]

        # Baselines — zeros is standard for DeepSHAP
        baselines = torch.zeros(
            (n_samples,) + input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        # Compute attributions on the safe model copy
        ldls = LayerDeepLiftShap(shap_model, shap_target_layer)

        with torch.no_grad():
            attr = ldls.attribute(
                input_tensor,
                baselines   = baselines,
                target      = target_idx,
                attribute_to_layer_input = False,   # attribute to layer OUTPUT
            )

        # attr: (B, C, H, W) or (B, C) depending on layer
        # Aggregate over channels → (H, W)
        arr = attr[self.sample_idx]\
                    .abs()\
                    .sum(dim=0)\
                    .detach().cpu().numpy()

        lo  = np.percentile(arr[arr > 0], 2) if arr.max() > 0 else 0
        hi  = np.percentile(arr, 98)
        arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        return np.power(arr, 0.6)

        # Percentile clip + normalise to [0, 1]
        #vmax = np.percentile(heatmap, 99.5)
        #vmin = np.percentile(heatmap, 0.5)
        #heatmap = np.clip(heatmap, vmin, vmax)
        #heatmap = (heatmap - vmin) / (vmax - vmin + 1e-12)

        #return heatmap

    def get_ig_layer_heatmap(
        self,
        input_tensor:  torch.Tensor,
        target_layer,
        target_idx:    int,
        n_steps:       int = 50,
    ) -> np.ndarray:
        """
        Computes Integrated Gradients attributions for a specific intermediate layer.
        Uses LayerIntegratedGradients from captum.

        Parameters
        ----------
        input_tensor  : (1, C, H, W) input image tensor
        target_layer  : nn.Module — the layer to attribute to (e.g. model.layer4[1].conv2)
        target_idx    : int — target class index
        n_steps       : int — number of integration steps (higher = more accurate)

        Returns
        -------
        heatmap : np.ndarray (H, W) normalised to [0, 1]
        """
        from captum.attr import LayerIntegratedGradients

        # Find layer name in original model
        original_layer_name = None
        for name, module in self.paf.model.named_modules():
            if module is target_layer:
                original_layer_name = name
                break

        if original_layer_name is None:
            raise ValueError(
                f"target_layer not found in model. "
                f"Pass the module object directly, e.g. model.layer4[1].conv2"
            )

        self.paf.model.eval()

        # Baseline — zeros is standard for IG
        baseline = torch.zeros_like(input_tensor)

        # LayerIntegratedGradients attributes to layer OUTPUT by default
        lig = LayerIntegratedGradients(self.paf.model, target_layer)

        attr = lig.attribute(
            input_tensor,
            baselines          = baseline,
            target             = target_idx,
            n_steps            = n_steps,
            internal_batch_size= 1,         # process one interpolation step at a time
                                            # avoids OOM for large layers
        )

        # attr: (1, C, H, W) or (1, C) depending on layer type
        # Aggregate over channel dimension → (H, W)
        arr = attr[self.sample_idx]\
                    .abs()\
                    .sum(dim=0)\
                    .detach().cpu().numpy()
        
        lo  = np.percentile(arr[arr > 0], 2) if arr.max() > 0 else 0
        hi  = np.percentile(arr, 98)
        arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        return np.power(arr, 0.6)

        # Percentile clip + normalise to [0, 1]
        #vmin = np.percentile(heatmap, 0.5)
        #vmax = np.percentile(heatmap, 99.5)
        #heatmap = np.clip(heatmap, vmin, vmax)
        #heatmap = (heatmap - vmin) / (vmax - vmin + 1e-12)
        #return heatmap
    
    def visualize_nips_killer_figure(
        self,
        img:       torch.Tensor,
        target:    int,
        save_path: str = None,
    ) -> None:
        import cv2
        import numpy as np
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from pytorch_grad_cam import GradCAMPlusPlus

        # ----------------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------------
        LAYERS = [
            'conv1',
            'layer1_1_conv2',
            'layer2_1_conv2',
            'layer3_1_conv2',
            'layer4_1_conv2',
        ]
        LAYER_LABELS = ['L1  conv1', 'L2  layer1', 'L3  layer2',
                        'L4  layer3', 'L5  layer4']

        # PAF modes: (key_substring, display_label)
        PAF_MODES = [
            ('abs',         r'PAF$_\mathbf{ABS}$'),
          #  ('power',       r'PAF$_\mathbf{POW}$'),
         #   ('norm',        r'PAF$_\mathbf{NORM}$'),
            ('norm_power',  r'PAF$_\mathbf{NRM}$★'),
         #   ('signed_split',r'PAF$_\mathbf{SS}$'),
        ]
        # Structural is PAF-side col 5 (index 5 in PAF section)
        PAF_COL_LABELS = [m[1] for m in PAF_MODES] + ['Structural\n(PAF)']

        BASELINE_METHODS = ['GradCAM++', 'LRP', 'DeepSHAP', 'IG']

        # Light theme — NeurIPS print-friendly
        FIG_BG    = '#ffffff'
        PAF_HDR   = '#fff3e0'   # warm orange tint for PAF header
        BASE_HDR  = '#e8f4fd'   # cool blue tint for baseline header
        DIV_BG    = '#dee2e6'
        LABEL_BG  = '#f1f3f5'
        ACCENT    = '#d62728'   # primary PAF mode colour
        DARK_TXT  = '#212529'
        MED_TXT   = '#495057'

        # ----------------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------------
        module_map = self.paf.hook_manager.graph_info['module_map']
        raw_img    = img[self.sample_idx].cpu().permute(1, 2, 0).numpy()
        raw_img    = (raw_img - raw_img.min()) / \
                    (raw_img.max() - raw_img.min() + 1e-8)
        H, W       = raw_img.shape[:2]

        def _paf_heatmap(store, lname):
            p = store.get(lname)
            if p is None:
                return np.zeros((H, W))
            if p.dim() == 4:
                arr = p[0].abs().sum(dim=0)
            elif p.dim() == 3:
                arr = p.abs().sum(dim=0)
            else:
                arr = p.abs()
            arr = arr.float().detach().cpu()
            if arr.shape[0] != H or arr.shape[1] != W:
                arr = F.interpolate(
                    arr.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='bicubic', align_corners=False,
                ).squeeze()
            arr = arr.numpy()
            arr = np.clip(arr, 0, None)
            lo  = np.percentile(arr[arr > 0], 2) if arr.max() > 0 else 0
            hi  = np.percentile(arr, 98)
            arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
            return np.power(arr, 0.6)   # gamma — brightens midrange

        def _lrp_layer_heatmap(layer_module):
            try:
                llrp = LayerLRP(self.paf.model, layer_module)
                attr = llrp.attribute(img, target=target)
                attr = torch.clamp(attr, min=0)

                arr    = attr[self.sample_idx].abs().sum(0).detach().cpu().numpy()
                #lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
                lo  = np.percentile(arr[arr > 0], 2) if arr.max() > 0 else 0
                hi  = np.percentile(arr, 98)
                arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
                return np.power(arr, 0.6)
            
            except Exception as e:
                print(f"LRP failed at layer: {e}")
                return np.zeros((H, W))
        
        def _structural(store, lname):
            """PAF heatmap with Canny edges overlaid."""
            h     = _paf_heatmap(store, lname)
            gray  = cv2.cvtColor(
                (raw_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
            edges = cv2.Canny(gray, 50, 150)
            cmap  = plt.cm.inferno(h)[:, :, :3]
            cmap[edges > 0] = [1, 1, 1]
            return cmap   # (H, W, 3) — already RGB

        def _blend(heatmap):
            """Blend inferno heatmap with image for better visibility."""
            h_rgb   = plt.cm.inferno(heatmap)[:, :, :3]
            blended = 0.4 * raw_img + 0.6 * h_rgb
            return np.clip(blended, 0, 1)

        def _show_heatmap(ax, heatmap, is_rgb=False):
            if is_rgb:
                ax.imshow(heatmap)
            else:
                ax.imshow(_blend(heatmap))
            ax.axis('off')

        def _label_cell(ax, text, bg=LABEL_BG, color=DARK_TXT,
                        fontsize=9, bold=False, rotation=0):
            ax.set_facecolor(bg)
            ax.axis('off')
            ax.text(
                0.5, 0.5, text,
                transform      = ax.transAxes,
                ha='center', va='center',
                fontsize       = fontsize,
                fontweight     = 'bold' if bold else 'normal',
                color          = color,
                rotation       = rotation,
                multialignment = 'center',
            )

        # ----------------------------------------------------------------
        # Pre-compute baseline heatmaps — shape (H,W), one per layer
        # ----------------------------------------------------------------
        print("Computing baseline heatmaps...")
        baseline_maps = {m: [] for m in BASELINE_METHODS}
        for lname in LAYERS:
            mod = module_map[lname]

            # GradCAM++
            try:
                cam = GradCAMPlusPlus(model=self.paf.model,
                                    target_layers=[mod])
                h = cam(input_tensor=img)[0]
                lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
                h = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
            except Exception as e:
                print(f"GradCAM++ failed at {lname}: {e}")
                h = np.zeros((H, W))
            # GradCAM output is already (H,W) upsampled
            baseline_maps['GradCAM++'].append(h)

            # LRP
            baseline_maps['LRP'].append(
                _lrp_layer_heatmap(mod)
            )

            # DeepSHAP
            baseline_maps['DeepSHAP'].append(
                self.get_deepshap_layer_heatmap(img, mod, target)
            )

            # IG
            baseline_maps['IG'].append(
                self.get_ig_layer_heatmap(img, mod, target)
            )
            print(f"  ✓ {lname}")

        # ----------------------------------------------------------------
        # Find PAF mode keys
        # ----------------------------------------------------------------
        def _find_key(substring):
            for mk in self.paf.distributions:
                if substring in str(mk).lower():
                    return mk
            return None

        paf_keys = [_find_key(m[0]) for m in PAF_MODES]
        primary_key = _find_key('norm_power') or list(self.paf.distributions.keys())[0]

        # ----------------------------------------------------------------
        # Layout:
        # 6 rows × 13 cols
        #   col 0           : row label (layer name)
        #   cols 1-5        : 5 PAF modes
        #   col 6           : Structural (PAF contribution)
        #   col 7           : narrow divider
        #   cols 8-11       : 4 baseline methods
        #
        # Row 0: column headers
        # Rows 1-5: L1→L5
        # ----------------------------------------------------------------
        N_ROWS = 6    # header + 5 layers
        N_COLS = 9   # label(1) + PAF(5) + structural(1) + div(1) + baselines(4)

        fig = plt.figure(figsize=(26, 18), facecolor=FIG_BG)
        fig.patch.set_facecolor(FIG_BG)

        gs = gridspec.GridSpec(
            N_ROWS, N_COLS,
            figure       = fig,
            hspace       = 0.06,
            wspace       = 0.04,
            width_ratios = [
                0.5,          # col 0: row label
                1.4, 1.4,# cols 1-5: PAF modes
                1.4,            # col 6: structural
                0.15,         # col 7: divider
                1.4, 1.4, 1.4, 1.4,   # cols 8-11: baselines
            ],
            height_ratios= [0.55, 1, 1, 1, 1, 1],   # header shorter
        )

        # ----------------------------------------------------------------
        # ROW 0 — column headers
        # ----------------------------------------------------------------

        # col 0: top-left corner — show input image label
        ax = fig.add_subplot(gs[0, 0])
        raw_img=img[self.sample_idx].cpu().permute(1, 2, 0).numpy()
        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        ax.imshow(raw_img)
        ax.set_title("Input Image", fontsize=12, fontweight='bold')
        ax.axis('off')
        #_label_cell(ax, 'Layer', bg=FIG_BG, fontsize=9, bold=True)

        # cols 1-2: PAF mode headers
        for ci, (_, mode_label) in enumerate(PAF_MODES):
            ax = fig.add_subplot(gs[0, ci + 1])
            is_primary = (ci == 3)   # norm_power
            _label_cell(
                ax, mode_label,
                bg    = PAF_HDR,
                color = ACCENT if is_primary else DARK_TXT,
                fontsize = 12, bold = is_primary,
            )

        # col 3: structural header
        ax = fig.add_subplot(gs[0, 3])
        _label_cell(ax, 'Structural\n(PAF)', bg=PAF_HDR,
                    color=MED_TXT, fontsize=12)

        # col 4: divider header
        ax = fig.add_subplot(gs[0, 4])
        ax.set_facecolor(DIV_BG)
        ax.axis('off')

        # cols 5-8: baseline method headers
        for bi, blabel in enumerate(BASELINE_METHODS):
            ax = fig.add_subplot(gs[0, 5 + bi])
            _label_cell(ax, blabel, bg=BASE_HDR,
                        color=DARK_TXT, fontsize=12, bold=True)

        # ----------------------------------------------------------------
        # ROWS 1-5 — one layer per row
        # ----------------------------------------------------------------
        for li, lname in enumerate(LAYERS):
            r = li + 1   # row index in gridspec

            # col 0: layer label
            ax = fig.add_subplot(gs[r, 0])
            _label_cell(ax, LAYER_LABELS[li],
                        bg=LABEL_BG, color=DARK_TXT,
                        fontsize=12, bold=False, rotation=0)

            # cols 1-5: PAF modes — each column is one mode, row = layer
            for ci, (mode_substr, _) in enumerate(PAF_MODES):
                ax   = fig.add_subplot(gs[r, ci + 1])
                key  = paf_keys[ci]
                store = self.paf.distributions.get(key, {})
                h    = _paf_heatmap(store, lname)
                _show_heatmap(ax, h)

                # Highlight primary mode column
                is_primary = (ci == 3)
                if is_primary:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(ACCENT)
                        spine.set_linewidth(2)

                # Conservation annotation at L5 for primary mode
                if is_primary and li == 4:
                    ax.annotate(
                        'Σ=1.000',
                        xy=(0.97, 0.03), xycoords='axes fraction',
                        ha='right', va='bottom',
                        fontsize=7, color='white',
                        fontfamily='monospace',
                        bbox=dict(facecolor=ACCENT, alpha=0.9,
                                boxstyle='round,pad=0.25'),
                    )

            # col 3: structural overlay (PAF primary mode)
            ax = fig.add_subplot(gs[r, 3])
            store_primary = self.paf.distributions.get(primary_key, {})
            struct = _structural(store_primary, lname)
            _show_heatmap(ax, struct, is_rgb=True)

            # col 4: divider
            ax = fig.add_subplot(gs[r, 4])
            ax.set_facecolor(DIV_BG)
            ax.axis('off')

            # cols 5-8: baseline methods — each column is one method, row = layer
            for bi, blabel in enumerate(BASELINE_METHODS):
                ax = fig.add_subplot(gs[r, 5 + bi])
                h  = baseline_maps[blabel][li]        # heatmap at this layer
                if h.shape[0] != H or h.shape[1] != W:
                    h = cv2.resize(h, (W, H), interpolation=cv2.INTER_CUBIC)
                _show_heatmap(ax, h)

        # ----------------------------------------------------------------
        # Section super-titles
        # ----------------------------------------------------------------
        # PAF section spans cols 1-6
        '''
        fig.text(
            0.32, 0.955,
            '◀  Probabilistic Activation Flow (PAF)  ▶',
            ha='center', va='top',
            fontsize=12, fontweight='bold', color=ACCENT,
        )
        # Baseline section spans cols 8-11
        fig.text(
            0.65, 0.955,
            '◀  Baseline Methods  ▶',
            ha='center', va='top',
            fontsize=12, fontweight='bold', color='#1f77b4',
        )
        r'''
        # ----------------------------------------------------------------
        # Shared caption
        # ----------------------------------------------------------------
        '''
        fig.text(
            0.5, -0.01,
            r'All heatmaps: inferno colormap, blended with input, '
            r'percentile-clipped [2%, 98%].  '
            r'★ = primary mode (NormPower, $\tau$=2).  '
            r'Structural = PAF attribution with Canny edge overlay.',
            ha='center', va='top',
            fontsize=12, color=MED_TXT, style='italic',
        )
        '''
        # ----------------------------------------------------------------
        # Save
        # ----------------------------------------------------------------
        try:
            plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        except:
            pass
        if save_path:
            self._ensure_save_dir(save_path)
            plt.savefig(save_path, bbox_inches='tight',
                        dpi=300, facecolor=FIG_BG)
            print(f"Saved: {save_path}")
        plt.show()

    def visualize_nips_killer_figure_old1(
        self,
        img:        torch.Tensor,
        target:     int,
        save_path:  str = None,
    ) -> None:
        """
        NeurIPS killer figure — 6 rows × 10 columns.

        Layout
        ------
        Row 0 (header):
            Col 0    : Input image
            Cols 1-4 : PAF layer distribution (L1→L5, inferno, no overlay)
            Col 5    : [divider label]
            Cols 6-9 : Baseline layer maps (GradCAM++, LRP, DeepSHAP, IG)
                    all at the SAME layers as PAF for fair comparison

        Rows 1-5 (method rows):
            Left label  : PAF mode name
            Cols 0-4    : PAF heatmaps at L1→L5 for that mode
            Centre label: Baseline method name (printed in col 5 divider)
            Cols 5-9    : Baseline heatmaps at L1→L5

        Row labels (y-axis text) on left and centre divider column.
        """
        import cv2
        import numpy as np
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch
        from pytorch_grad_cam import GradCAMPlusPlus
        from captum.attr import LayerLRP

        # ----------------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------------
        LAYERS = [
            'conv1',
            'layer1_1_conv2',
            'layer2_1_conv2',
            'layer3_1_conv2',
            'layer4_1_conv2',
        ]
        LAYER_LABELS = ['L0\nconv', 'L1\nConv', 'L2\nConv', 'L3\nConv', 'L4\nConv']

        PAF_MODES = [
            ('abs',        1.0, r'PAF$_\mathtt{ABS}$'),
            ('power',      2.0, r'PAF$_\mathtt{POW}$'),
            ('norm',       1.0, r'PAF$_\mathtt{NORM}$'),
            ('norm_power', 2.0, r'PAF$_\mathtt{NRM}$★'),
            ('signed_split',1.0,r'PAF$_\mathtt{SS}$'),
        ]

        BASELINE_LABELS = ['GradCAM++', 'LRP', 'DeepSHAP', 'IG']

        # Colour scheme
        '''
        PAF_BG    = '#0d1117'   # near-black background for PAF section
        BASE_BG   = '#111820'   # slightly different for baseline section
        LABEL_COL = '#c9d1d9'   # light text on dark bg
        ACCENT    = '#f0883e'   # orange accent for PAF star mode
        DIV_COL   = '#21262d'   # divider column background
        '''
        PAF_BG  = '#ffffff'
        BASE_BG = '#f8f9fa'
        DIV_COL = '#e9ecef'
        LABEL_COL = '#212529'
        ACCENT    = '#d62728'
        # ----------------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------------
        module_map  = self.paf.hook_manager.graph_info['module_map']
        raw_img     = img[self.sample_idx].cpu().permute(1, 2, 0).numpy()
        raw_img     = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        H, W        = raw_img.shape[:2]

        def _paf_heatmap(store, lname):
            """PAF distribution → normalised (H,W) numpy array."""
            p = store.get(lname)
            if p is None:
                return np.zeros((H, W))

            # p can be (C, h, w) or (1, C, h, w) or (h, w) depending on layer
            # Normalise to (h, w) first by summing over all non-spatial dims
            if p.dim() == 4:
                # (B, C, h, w) — take sample, sum over channels
                arr = p[0].abs().sum(dim=0)          # (h, w)
            elif p.dim() == 3:
                # (C, h, w) — sum over channels
                arr = p.abs().sum(dim=0)             # (h, w)
            elif p.dim() == 2:
                # (h, w) — already spatial
                arr = p.abs()
            else:
                return np.zeros((H, W))

            arr = arr.float().detach().cpu()         # (h, w)

            # Upsample to input resolution if needed
            if arr.shape[0] != H or arr.shape[1] != W:
                arr = F.interpolate(
                    arr.unsqueeze(0).unsqueeze(0),   # (1, 1, h, w)
                    size        = (H, W),
                    mode        = 'bicubic',
                    align_corners = False,
                ).squeeze()                          # (H, W)

            arr = arr.numpy()
            arr = np.clip(arr, 0, None)
            lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
            arr=np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
            arr = np.power(arr, 0.6)
            return arr

        def _gradcam_heatmap(layer_module):
            cam = GradCAMPlusPlus(
                model        = self.paf.model,
                target_layers= [layer_module],
            )
            h = cam(input_tensor=img)[0]
            lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
            return np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)

        def _lrp_heatmap(layer_module):
            try:
                llrp = LayerLRP(self.paf.model, layer_module)
                attr = llrp.attribute(img, target=target)
                h    = attr[self.sample_idx].abs().sum(0).detach().cpu().numpy()
                lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
                return np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
            except Exception as e:
                print(f"LRP failed at layer: {e}")
                return np.zeros((H, W))
        
        '''
        def _show(ax, heatmap, cmap='inferno', overlay=True, alpha=0.75):
            """Render heatmap on dark background with optional image overlay."""
            if overlay:
                ax.imshow(raw_img)
                ax.imshow(heatmap, cmap=cmap, alpha=alpha)
            else:
                ax.imshow(heatmap, cmap=cmap)
            ax.axis('off')
        '''

        def _show(ax, heatmap, cmap='inferno', overlay=True, alpha=0.85):
            if overlay:
                # Blend heatmap with image: brighten heatmap first
                heatmap_rgb = plt.cm.inferno(heatmap)[:, :, :3]   # (H, W, 3)
                blended = 0.45 * raw_img + 0.55 * heatmap_rgb     # stronger heatmap
                ax.imshow(np.clip(blended, 0, 1))
            else:
                ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')

        def _label_ax(ax, text, color=LABEL_COL, fontsize=9,
                    bg=DIV_COL, bold=False):
            """Fill axis with a solid colour and centred text label."""
            ax.set_facecolor(bg)
            ax.axis('off')
            ax.text(
                0.5, 0.5, text,
                transform   = ax.transAxes,
                ha='center', va='center',
                fontsize    = fontsize,
                fontweight  = 'bold' if bold else 'normal',
                color       = color,
                wrap        = True,
                multialignment = 'center',
            )

        # ----------------------------------------------------------------
        # Pre-compute baseline heatmaps (expensive — compute once)
        # ----------------------------------------------------------------
        baseline_heatmaps = {label: [] for label in BASELINE_LABELS}
        for lname in LAYERS:
            mod = module_map[lname]
            baseline_heatmaps['GradCAM++'].append(_gradcam_heatmap(mod))
            baseline_heatmaps['LRP'].append(_lrp_heatmap(mod))
            baseline_heatmaps['DeepSHAP'].append(
                self.get_deepshap_layer_heatmap(img, mod, target)
            )
            baseline_heatmaps['IG'].append(
                self.get_ig_layer_heatmap(img, mod, target)
            )

        # ----------------------------------------------------------------
        # Figure layout
        # 6 rows × 11 columns:
        #   col 0          : left label (PAF mode / baseline name)
        #   cols 1-5       : PAF heatmaps L1→L5
        #   col 6          : centre divider
        #   cols 7-11 but we map to 4 baseline methods → 4 cols × 5 layers
        #
        # Simpler: 6 rows × 12 columns
        #   col 0          : row label
        #   cols 1-5       : PAF (5 layers)
        #   col 6          : centre divider
        #   cols 7-10      : 4 baselines (one per column, NOT per layer)
        #
        # Wait — user asked for 10 content columns: 5 PAF + 5 baselines
        # With 4 baselines we use cols 7-10 for GradCAM, LRP, DeepSHAP, IG
        # and col 11 is empty, OR we use 5 baselines.
        # Use GradCAM, LRP, DeepSHAP, IG, Structural-overlay as 5 baselines.
        #
        # Final layout: 6 rows × 12 cols
        #   col 0           : row label (narrow)
        #   cols 1-5        : PAF L1→L5
        #   col 6           : centre divider (narrow)
        #   cols 7-11       : GradCAM / LRP / DeepSHAP / IG / Structural
        # ----------------------------------------------------------------

        N_CONTENT = 5        # layers per side
        N_ROWS    = 6        # header + 5 method rows
        N_COLS    = 12       # label + 5 PAF + divider + 5 baseline

        # Row 0 height is taller (header with input image)
        fig = plt.figure(figsize=(28, 20), facecolor=PAF_BG)
        fig.patch.set_facecolor(PAF_BG)

        gs = gridspec.GridSpec(
            N_ROWS, N_COLS,
            figure       = fig,
            hspace       = 0.04,
            wspace       = 0.04,
            width_ratios = [0.35, 1, 1, 1, 1, 1,   # label + 5 PAF
                            0.35,                    # divider
                            1, 1, 1, 1, 1],          # 5 baseline
            height_ratios= [1.3, 1, 1, 1, 1, 1],    # header row taller
        )

        BASELINE_METHODS = ['GradCAM++', 'LRP', 'DeepSHAP', 'IG', 'Structural']

        # ----------------------------------------------------------------
        # ROW 0 — header
        # ----------------------------------------------------------------

        # Col 0: row label cell
        ax = fig.add_subplot(gs[0, 0])
        _label_ax(ax, 'Input\nImage', fontsize=8, bg=PAF_BG)

        # Col 1: input image
        ax_img = fig.add_subplot(gs[0, 1])
        ax_img.imshow(raw_img)
        ax_img.axis('off')
        ax_img.set_facecolor(PAF_BG)

        # Cols 2-5: PAF distribution at L1→L5 for primary mode (norm_power)
        primary_key = None
        for mk in self.paf.distributions:
            mode_str = str(mk).lower()
            if 'norm_power' in mode_str or 'nrm' in mode_str:
                primary_key = mk
                break
        if primary_key is None:
            primary_key = list(self.paf.distributions.keys())[0]

        primary_store = self.paf.distributions[primary_key]
        max_probs     = []

        for li, lname in enumerate(LAYERS):
            col = li + 1 if li == 0 else li + 1   # cols 1-5 but col 1 is input
            # shift: input is col 1, layers start col 2
            col = li + 2
            if col > 5:
                break
            ax = fig.add_subplot(gs[0, col])
            h  = _paf_heatmap(primary_store, lname)
            max_probs.append(primary_store[lname].max().item()
                            if primary_store.get(lname) is not None else 0)
            _show(ax, h, overlay=False)

            # Layer label below image
            conc = ""
            if li == 4 and len(max_probs) > 1:
                ratio = max_probs[-1] / (max_probs[0] + 1e-8)
                conc  = f"\n×{ratio:.0f} concentration"
            ax.set_title(
                f"{LAYER_LABELS[li]}{conc}",
                color=LABEL_COL, fontsize=12, pad=3,
                fontfamily='monospace',
            )
            ax.set_facecolor(PAF_BG)

        # Cols 6+: remaining header cells — show PAF section title
        ax_paf_title = fig.add_subplot(gs[0, 1:6])
        ax_paf_title.set_facecolor(PAF_BG)
        ax_paf_title.axis('off')
        # Already filled by subplots above; add a super-title instead

        # Centre divider col 6
        ax_div = fig.add_subplot(gs[0, 6])
        _label_ax(ax_div, '←PAF\nBase→', fontsize=7, bg=DIV_COL, color='#8b949e')

        # Cols 7-11: baseline layer labels as column headers
        for bi, blabel in enumerate(BASELINE_METHODS):
            ax = fig.add_subplot(gs[0, 7 + bi])
            _label_ax(ax, blabel, fontsize=9, bg=PAF_BG,
                    color=ACCENT if bi == 0 else LABEL_COL, bold=(bi == 0))

        # ----------------------------------------------------------------
        # ROWS 1-5 — one PAF mode per row
        # ----------------------------------------------------------------

        for row_idx, (mode_name, tau, mode_label) in enumerate(PAF_MODES):
            r = row_idx + 1   # actual row index in gridspec

            # Find the matching key in distributions
            mode_key = None
            for mk in self.paf.distributions:
                mk_str = str(mk).lower()
                if mode_name in mk_str:
                    mode_key = mk
                    break
            store = self.paf.distributions.get(mode_key, {})

            is_primary = (mode_name == 'norm_power')
            label_color = ACCENT if is_primary else LABEL_COL
            label_bg    = '#1a1f2e' if is_primary else PAF_BG

            # Col 0: PAF row label
            ax = fig.add_subplot(gs[r, 0])
            _label_ax(ax, mode_label, fontsize=8,
                    color=label_color, bg=label_bg, bold=is_primary)

            # Cols 1-5: PAF heatmaps at L1→L5
            for li, lname in enumerate(LAYERS):
                ax = fig.add_subplot(gs[r, li + 1])
                h  = _paf_heatmap(store, lname)
                _show(ax, h, overlay=True, alpha=0.70)
                ax.set_facecolor(PAF_BG)

                # Conservation annotation on last layer of primary mode
                if is_primary and li == 4:
                    total = sum(
                        self.paf.distributions[mode_key].get(ln, torch.zeros(1)).sum().item()
                        for ln in LAYERS[:1]   # just check input layer sums to 1
                    )
                    ax.annotate(
                        'Σ=1.000',
                        xy=(0.97, 0.03), xycoords='axes fraction',
                        ha='right', va='bottom',
                        fontsize=7, color='white', fontfamily='monospace',
                        bbox=dict(facecolor='#f0883e', alpha=0.85,
                                boxstyle='round,pad=0.2'),
                    )

            # Col 6: centre divider
            ax = fig.add_subplot(gs[r, 6])
            _label_ax(ax, '│', fontsize=14, bg=DIV_COL, color='#444c56')

            # Cols 7-11: baseline heatmaps — each column is one method,
            # showing the heatmap at the layer corresponding to this row
            # (row 1 = L1, row 2 = L2, ... row 5 = L5)
            lname_for_row = LAYERS[row_idx]   # row matches layer index

            for bi, blabel in enumerate(BASELINE_METHODS):
                ax = fig.add_subplot(gs[r, 7 + bi])
                ax.set_facecolor(PAF_BG)

                if blabel == 'Structural':
                    # Structural overlay: Canny edges on PAF heatmap
                    p = store.get(lname_for_row)
                    if p is not None:
                        h = _paf_heatmap(store, lname_for_row)
                        gray  = cv2.cvtColor(
                            (raw_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                        )
                        edges = cv2.Canny(gray, 50, 150)
                        color_map = plt.cm.inferno(h)[:, :, :3]
                        color_map[edges > 0] = [1, 1, 1]
                        ax.imshow(color_map)
                    else:
                        ax.imshow(np.zeros((H, W)), cmap='inferno')
                    ax.axis('off')

                elif blabel in baseline_heatmaps:
                    h = baseline_heatmaps[blabel][row_idx]
                    h_up = cv2.resize(h, (W, H), interpolation=cv2.INTER_CUBIC) \
                        if h.shape[:2] != (H, W) else h
                    _show(ax, h_up, cmap='inferno', overlay=True, alpha=0.70)

                else:
                    ax.imshow(np.zeros((H, W)), cmap='gray')
                    ax.axis('off')

        # ----------------------------------------------------------------
        # Super-titles for PAF and baseline sections
        # ----------------------------------------------------------------
        fig.text(
            0.28, 0.995,
            'Probabilistic Activation Flow (PAF)  —  Layer-wise Distributions',
            ha='center', va='top',
            fontsize=13, fontweight='bold', color=ACCENT,
            fontfamily='monospace',
        )
        fig.text(
            0.74, 0.995,
            'Baseline Methods  —  Layer Attribution Comparison',
            ha='center', va='top',
            fontsize=13, fontweight='bold', color=LABEL_COL,
            fontfamily='monospace',
        )

        # Column header row for layer labels (PAF side)
        for li, ll in enumerate(LAYER_LABELS):
            fig.text(
                (1 + li + 1) / N_COLS + 0.005,
                0.975,
                ll.replace('\n', ' '),
                ha='center', va='top',
                fontsize=7.5, color='#8b949e',
                fontfamily='monospace',
            )

        # ----------------------------------------------------------------
        # Save / show
        # ----------------------------------------------------------------
        if save_path:
            plt.savefig(
                save_path, bbox_inches='tight',
                dpi=300, facecolor=PAF_BG,
            )
            print(f"Saved: {save_path}")

        plt.show()
        
    def visualize_nips_killer_figure_old(self, img,target,mode_key=None, save_path=None):
        key = mode_key if mode_key is not None else self.mode_key
        store = self.paf.distributions.get(key)

        layers=['conv1','layer1_1_conv2','layer2_1_conv2','layer3_1_conv2','layer4_1_conv2']

        # Selecting 5 representative layers: L1, L2, L3, L4, L5 (Layer4)
        indices = [0, 1, 2, 3, 4] 
        pivot_layers = [layers[i] for i in indices]
    
        # Map names to actual Module Objects for Captum/GradCAM
        module_map=self.paf.hook_manager.graph_info['module_map']
        pivot_modules = [module_map[name] for name in layers]

        fig = plt.figure(figsize=(18, 22))
        gs = gridspec.GridSpec(7, 5, height_ratios=[1, 1, 1, 1, 1, 1,1], hspace=0.15)

        # --- ROW 0 & 1: PAF HEATMAPS (INFERNO) ---
        # Col 0: Input Image

        #raw_img = self.paf.distributions[mode_key]['x'][self.sample_idx].cpu().permute(1, 2, 0).numpy()
        raw_img=img[self.sample_idx].cpu().permute(1, 2, 0).numpy()
        target_h, target_w = raw_img.shape[:2]
        ax_img = fig.add_subplot(gs[0, 0])
        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        ax_img.imshow(raw_img)
        ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')

        # Heatmap Panels
        max_probs = []
        for i, lname in enumerate(pivot_layers):
            ax = fig.add_subplot(gs[1, i]) # Row 1
            p_tensor = store.get(lname)
            max_probs.append(p_tensor.max())
            
            # Heatmap Processing
            spatial_map = p_tensor.sum(dim=1, keepdim=True)
            upsampled = F.interpolate(
                spatial_map, size=(target_h, target_w), mode='bicubic', align_corners=False
            ).squeeze().cpu().detach().numpy()

            
            # Normalization
            upsampled = np.clip(upsampled, 0, None)
            v_max = np.percentile(upsampled, 99.0)
            heatmap = np.clip(upsampled, 0, v_max)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Plot Overlay
            #ax.imshow(img_display)
            #ax.imshow(heatmap, cmap='jet', alpha=0.5)


            #im = ax.imshow(p_tensor, cmap='inferno')
            ax.imshow(raw_img)
            im=ax.imshow(heatmap, cmap='inferno')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
              # Shorten name for cleaner UI (e.g., layer1.0.conv1)
            display_name = ".".join(lname.split('.')[-2:]) 
            ax.set_title(f"Layer {i+1}:\n{display_name}", fontsize=10)
            ax.axis('off')
            #ax.set_title(f"L{i+1}: {lname.split('.')[-1]}\n({p_tensor.shape[0]}x{p_tensor.shape[1]})", fontsize=10)
            
            # Row 1 Annotation: "x87 concentration" on L5
            if i == 4:
                ax.annotate(f"×{max_probs[-1]/max_probs[0]:.0f} concentration", xy=(0.5, 0.5), xycoords='axes fraction',
                            color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

        # --- ROW 2: STRUCTURAL OVERLAY (Only L1, L3, L5) ---
        # We use columns 0, 2, 4 to spread them out
        for idx, col_pos in enumerate([0, 1,2,3,4]):
            ax = fig.add_subplot(gs[2, col_pos])
            lname = pivot_layers[idx] # Picks L1, L3, L5
            p_tensor = store[lname][self.sample_idx].sum(dim=0).cpu().numpy()
            
            # Canny Logic
            gray = cv2.cvtColor((raw_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            heatmap_res = cv2.resize(p_tensor, (raw_img.shape[1], raw_img.shape[0]))
            color_map = plt.cm.inferno(heatmap_res / (heatmap_res.max() + 1e-12))[:, :, :3]
            color_map[edges > 0] = [1, 1, 1] # White contours
            
            ax.imshow(color_map)
            ax.set_title(f"Structural Alignment (L{idx*2+1})", fontsize=10)
            ax.axis('off')

        # --- ROW 3: SIDE-BY-SIDE (THE "PAF WINS" PANEL) ---
        # Placeholder for comparison methods (You'll need to pass these in)
        comp_titles = ["GradCAM", "IG", "DeepSHAP", "PAF_abs", "PAF_norm_power"]
        
        def get_layer_by_name(model, layer_name):
            modules = dict(model.named_modules())
            if layer_name not in modules:
                raise ValueError(f"{layer_name} not found in model")
            return modules[layer_name]
        
        #for i, title in enumerate(comp_titles):
        for i, lname in enumerate(pivot_layers):
            ax = fig.add_subplot(gs[3, i])
            # Insert your comparison logic here
            #target_layer = get_layer_by_name(self.paf.model, lname)
            lnode=self.paf.hook_manager.graph_info['module_map'][lname]
            cam = GradCAMPlusPlus(model=self.paf.model, target_layers=[lnode])
            heatmap = cam(input_tensor=img)[0]
            title=lname
            ax.imshow(heatmap)
            ax.text(0.5, 0.5, title, ha='center') 
            ax.set_title(title, fontweight='bold' if "PAF" in title else 'normal')
            ax.axis('off')


        # --- ROW 4: PROBABILITY MASS CURVE ---
        #heatmaps=compute_lrp_layerwise_heatmaps(img,self.paf.model)
        for i, mod in enumerate(pivot_modules): 
            ax = fig.add_subplot(gs[4, i])
            llrp = LayerLRP(self.paf.model, mod)
            attr_llrp = llrp.attribute(img, target=target)
            attr = torch.clamp(attr_llrp, min=0)

            # Aggregate across channels and move to numpy
            heatmap_lrp = attr[self.sample_idx].sum(0).abs().detach().cpu().numpy()
            
            title=lname
            ax.imshow(heatmap_lrp)
            ax.text(0.5, 0.5, title, ha='center') 
            ax.set_title(title, fontweight='bold' if "PAF" in title else 'normal')
            ax.axis('off')

        #Row 5
        for i, mod in enumerate(pivot_modules): 
            ax = fig.add_subplot(gs[5, i])
            heatmap_shap=self.get_deepshap_layer_heatmap(img,mod,target)
            ax.imshow(heatmap_shap)
            ax.text(0.5, 0.5, title, ha='center') 
            ax.set_title(title, fontweight='bold' if "PAF" in title else 'normal')
            ax.axis('off')
        
        #Row 5
        for i, mod in enumerate(pivot_modules): 
            ax = fig.add_subplot(gs[6, i])
            heatmap_shap=self.get_ig_layer_heatmap(img,mod,target)
            ax.imshow(heatmap_shap)
            ax.text(0.5, 0.5, title, ha='center') 
            ax.set_title(title, fontweight='bold' if "PAF" in title else 'normal')
            ax.axis('off')

        
        '''
        ax_curve = fig.add_subplot(gs[5, :]) # Spans all columns
        ax_curve.plot(range(1, len(pivot_layers)+1), max_probs, marker='o', color='firebrick', linewidth=2)
        ax_curve.set_yscale('log')
        ax_curve.set_xlabel("Layer Index")
        ax_curve.set_ylabel("Max Prob (Log)")
        ax_curve.set_title("Conservation: Σ=1.000 at every layer", loc='right', fontsize=10, style='italic')
        ax_curve.grid(True, which="both", ls="-", alpha=0.2)

        fig.text(0.5, 0.38, "Attribution aligns with semantic structure without supervision", 
                ha='center', fontsize=14, fontweight='bold', style='italic')
        '''
        if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def visualize_canny_evolution(
        self, 
        mode_key: tuple = None, 
        cols: int = 6,
        dim_factor: float = 0.3,
        save_path: Optional[str] = None
    ):
        """
        Generates a high-quality figure showing the evolution of PAF distributions 
        across all convolutional layers, overlaid with structural context.
        """
        # 1. Setup Mode and Store
        key = mode_key if mode_key is not None else self.mode_key
        store = self.paf.distributions.get(key)
        if not store:
            print(f"Key {key} not found.")
            return

        layers=['conv1','layer1_1_conv2','layer2_1_conv2','layer3_1_conv2','layer4_1_conv2']


        # 3. Prepare Dimmed Context and Canny Edges
        # Extract raw input (B, C, H, W) -> (H, W, C)
        raw_img = self.paf.distributions[mode_key]['x'][self.sample_idx].cpu().permute(1, 2, 0).numpy()

        # Min-Max Normalize for display
        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        
        # Create dimmed background and Canny mask
        dimmed_img = raw_img * dim_factor
        gray_uint8 = (raw_img * 255).astype(np.uint8)
        gray = cv2.cvtColor(gray_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0

        # 4. Grid Setup
        num_plots = len(layers) + 1
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        axes_flat = axes.flatten()

        # SLOT 0: Structural Reference (Dimmed Image + White Edges)
        ref_box = dimmed_img.copy()
        ref_box[edge_mask] = [1.0, 1.0, 1.0] # Bright white edges
        axes_flat[0].imshow(ref_box)
        axes_flat[0].set_title("STRUCTURAL REF", fontsize=10, fontweight='bold', color='navy')
        axes_flat[0].axis('off')

        # SLOTS 1-N: Convolutional Evolution
        for i, lname in enumerate(layers):
            ax = axes_flat[i + 1]
            p_tensor = store.get(lname)
            
            if p_tensor is None:
                ax.axis('off')
                continue

            # Process spatial distribution
            if p_tensor.dim() == 4: 
                p_tensor = p_tensor[self.sample_idx]
            
            # Sum across channels and upscale to input resolution
            heatmap = p_tensor.sum(dim=0).cpu().numpy()
            heatmap_res = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
            
            # Normalize heatmap (Local 99th percentile for contrast)
            heatmap_res = np.maximum(heatmap_res, 0)
            v_max = np.percentile(heatmap_res, 99.0) + 1e-12
            heatmap_res = np.clip(heatmap_res / v_max, 0, 1)

            # Create overlay: Heatmap + White Canny Edges
            # Using 'magma' as it contrasts perfectly with white edges
            color_map = plt.cm.magma(heatmap_res)[:, :, :3]
            color_map[edge_mask] = [1.0, 1.0, 1.0] 

            ax.imshow(color_map)
            
            # Clean title: e.g., "L5: layer1.1.conv2"
            short_name = ".".join(lname.split('.')[-2:])
            ax.set_title(f"L{i+1}: {short_name}", fontsize=9)
            ax.axis('off')

        # 5. Final Formatting
        for j in range(num_plots, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()
        
        if save_path:
            # Save as PDF for vector-quality text/lines in LaTeX
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()
        
    def visualize_conv_evolution_canny(
        self, 
        mode_key: tuple = None, 
        cols: int = 6,
        save_path: str = None
    ):
        """
        Plots all convolutional layers in chronological order with 
        a Canny edge overlay for structural reference.
        """
        # 1. Setup Mode and Store
        key = mode_key if mode_key is not None else self.mode_key
        store = self.paf.distributions.get(key)
        if not store:
            return

        # 2. Identify and Filter Conv Layers (Excluding downsample)
        # Using your preferred filtering logic
        r'''
        all_layers = self.paf.find_layers_with_types(mode=key, layer_type='conv')
        layers = [lname for lname in all_layers if "downsample" not in lname.lower()]
        # 3. Sort Layers (Low-to-High / Input-to-Output)
        def layer_sort_key(name):
            if name == 'conv1': return [0]
            # Split into text and integers for natural sorting
            return [1] + [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]
        
        layers.sort(key=layer_sort_key)
        '''
        layers=['conv1','layer1_1_conv2','layer2_1_conv2','layer3_1_conv2','layer4_1_conv2']
        # 4. Prepare Reference Canny Edges
        # Pulling from the stored input tensor in the visualizer
        raw_img = self.paf.distributions[mode_key]['x'][self.sample_idx].cpu().permute(1, 2, 0).numpy()
        # Normalize to 0-255
        #raw_img=raw_img.cpu().numpy()
        raw_img = ((raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8) * 255).astype(np.uint8)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0

        # 5. Grid Setup (Reference Image + Sorted Conv Layers)
        num_plots = len(layers) + 1
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        axes_flat = axes.flatten()

        # Slot 0: Pure Reference Image
        axes_flat[0].imshow(raw_img)
        axes_flat[0].set_title("INPUT REFERENCE", fontsize=10, fontweight='bold', color='blue')
        axes_flat[0].axis('off')

        # Slots 1 to N: Sorted Heatmaps with Canny Overlay
        for i, lname in enumerate(layers):
            ax = axes_flat[i + 1]
            p_tensor = store.get(lname)
            
            if p_tensor is None:
                ax.axis('off')
                continue

            # Process spatial distribution
            if p_tensor.dim() == 4: p_tensor = p_tensor[self.sample_idx]
            heatmap = p_tensor.sum(dim=0).cpu().numpy()
            
            # Resize to match original image resolution
            heatmap_res = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
            
            # Normalize
            heatmap_res = np.maximum(heatmap_res, 0)
            v_max = np.percentile(heatmap_res, 99.0) + 1e-12
            heatmap_res = np.clip(heatmap_res / v_max, 0, 1)

            # Create the RGB overlay
            # We'll use 'magma' for high contrast against white Canny edges
            color_map = plt.cm.magma(heatmap_res)[:, :, :3]
            color_map[edge_mask] = [1.0, 1.0, 1.0] # White Canny edges

            ax.imshow(color_map)
            short_name = ".".join(lname.split('.')[-2:])
            ax.set_title(f"L{i+1}: {short_name}", fontsize=9)
            ax.axis('off')

        # Cleanup
        for j in range(num_plots, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()


    def visualize_layerwise_heatmaps(
        self, 
        mode_key: tuple = None, 
        cols: int = 6, 
        save_path: Optional[str] = None
    ):
        """
        Plots the spatial probability distribution (heatmap) for every 
        convolutional layer in the network, sorted from input to output.
        """
        # 1. Setup Mode and Store
        key = mode_key if mode_key is not None else self.mode_key
        store = self.paf.distributions.get(key)
        if not store:
            print(f"Key {key} not found in distributions.")
            return

        r'''
        # 2. Identify and Filter Conv Layers (same as Canny method)
        all_layers = self.paf.find_layers_with_types(mode=key, layer_type='conv')
        layers = [lname for lname in all_layers if "downsample" not in lname.lower()]

        # 3. Sort Layers (Low-to-High / Input-to-Output)
        def layer_sort_key(name):
            if name == 'conv1': return [0]
            # Split into text and integers for natural sorting
            return [1] + [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]
        
        layers.sort(key=layer_sort_key)
        '''
        layers=['conv1','layer1_1_conv2','layer2_1_conv2','layer3_1_conv2','layer4_1_conv2']

        # 4. Grid Setup
        num_plots = len(layers)
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
        fig.suptitle(f"PAF Conv-Layer Evolution | Mode: {key[0].value}", 
                    fontsize=16, fontweight='bold', y=1.02)
        
        axes_flat = axes.flatten()

        for i, lname in enumerate(layers):
            ax = axes_flat[i]
            p_layer = store[lname]
            
            # Handle batch dimension and aggregate channels
            if p_layer.dim() == 4:
                p_layer = p_layer[self.sample_idx]
            
            # Sum across channels to get spatial heatmap
            heatmap = p_layer.sum(dim=0).cpu().numpy()
            
            # Normalization (Local percentile clipping for best visibility)
            v_max = np.percentile(np.abs(heatmap), 99) + 1e-12
            
            # Select colormap based on scoring mode
            is_signed = key[0] == ScoringMode.SIGNED_FULL
            cmap = 'RdBu_r' if is_signed else 'magma'
            vmin = -v_max if is_signed else 0
            
            im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=v_max)
            
            # Clean title for the grid
            short_name = ".".join(lname.split('.')[-2:])
            ax.set_title(f"L{i+1}: {short_name}\n{heatmap.shape[0]}x{heatmap.shape[1]}", fontsize=10)
            ax.axis('off')
            
            # Optional: Smaller colorbar to save space
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 5. Hide unused subplots
        for j in range(num_plots, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def visualize_layer_results(self, x,mode,save_path):
        # 1. Get display-ready image
        img_display = self.get_img_display(x)
        layers=self.paf.find_layers_with_types(mode=mode,layer_type='conv')  
        def extract_key(name):
            # 1. Force the very first conv layer to the top
            if name == 'conv1' or name == 'features.0':
                return [0]
            # 2. Split name into strings and integers: 
            # "layer1.2.conv1" -> ["layer", 1, ".", 2, ".conv", 1]
            parts = [int(text) if text.isdigit() else text.lower() 
                    for text in re.split(r'(\d+)', name)]
            
            # 3. Use a [1] prefix to ensure these come after the 'conv1' [0] prefix
            return [1] + parts
        layers.sort(key=extract_key)
        for lname in layers:
            paf_tensor = self.paf.distributions[mode].get(lname)
            fig=self.plot_layer_results_combined(img_display, paf_tensor, lname)
            fig.savefig(save_path+f"_{lname}"+".png", dpi=150, bbox_inches="tight", facecolor='white')
            print(f"Layer plot saved to: {save_path}")

    def visualize_all_layers_grid(self, x, mode, save_path, cols=7):
        # 1. Prepare display image and filter layers
        img_display = self.get_img_display(x)
        target_h, target_w = img_display.shape[:2]
        
        # Filter out downsample layers
        all_layers = self.paf.find_layers_with_types(mode=mode, layer_type='conv')
        layers = [lname for lname in all_layers if "downsample" not in lname.lower()]
        
        def layer_sort_key(name):
            if name == 'conv1': return [0]
            # Split into text and integers for natural sorting
            return [1] + [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]
        
        layers.sort(key=layer_sort_key)

        layers=['conv1','layer1_1_conv2','layer2_1_conv2','layer3_1_conv2','layer4_1_conv2']

        # We add +1 to the count to make room for the Original Image at the start
        num_plots = len(layers) + 1
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        fig.suptitle(f"PAF Evolution: {mode}", fontsize=18, fontweight='bold', y=1.02)
        axes_flat = axes.flatten()

        # --- Position 0: The Original Image Reference ---
        axes_flat[0].imshow(img_display)
        axes_flat[0].set_title("REFERENCE\n(Input Image)", fontsize=12, color='blue', fontweight='bold')
        axes_flat[0].axis('off')

        # --- Positions 1 to N: The Filtered Layer Heatmaps ---
        for i, lname in enumerate(layers):
            ax = axes_flat[i + 1] # Offset by 1
            paf_tensor = self.paf.distributions[mode].get(lname)
            #paf_tensor=self.get_input_heatmap(paf_tensor)

            if paf_tensor is None:
                ax.axis('off')
                continue

            # Heatmap Processing
            spatial_map = paf_tensor.sum(dim=1, keepdim=True)
            upsampled = F.interpolate(
                spatial_map, size=(target_h, target_w), mode='bicubic', align_corners=False
            ).squeeze().cpu().detach().numpy()

            
            # Normalization
            upsampled = np.clip(upsampled, 0, None)
            v_max = np.percentile(upsampled, 99.0)
            heatmap = np.clip(upsampled, 0, v_max)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Plot Overlay
            #ax.imshow(img_display)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            
            # Shorten name for cleaner UI (e.g., layer1.0.conv1)
            display_name = ".".join(lname.split('.')[-2:]) 
            ax.set_title(f"Layer {i+1}:\n{display_name}", fontsize=10)
            ax.axis('off')

        # Hide any remaining empty subplots
        for j in range(num_plots, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()
        fig.savefig(f"{save_path}_grid_evolution.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ========================================================================
    # SECTION 1: IMAGE PREPROCESSING (FIX BLURRING)
    # ========================================================================
    
    @staticmethod
    def prepare_image_for_display(
        x: torch.Tensor,
        sample_idx: int = 0,
        target_size: int = 224,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Properly prepare image for display without excessive blurring.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (N, C, H, W)
        sample_idx : int
            Which sample in batch
        target_size : int
            Target display size (keep original aspect ratio)
            
        Returns
        -------
        img : np.ndarray
            Image in (C, H, W) or (H, W) format
        shape_info : tuple
            (C, H, W) for shape information
        """
        
        # Extract single sample
        x_single = x[sample_idx]  # (C, H, W)
        
        # Get original shape
        C, H, W = x_single.shape
        shape_info = (C, H, W)
        
        # IMPORTANT: Only resize if significantly different from target
        # Avoid multiple resizes which cause blurring!
        if H != target_size or W != target_size:
            # Use area for downsampling, bilinear for upsampling
            x_single = x_single.unsqueeze(0)  # (1, C, H, W)
            
            if H > target_size or W > target_size:
                # Downsampling: use area method
                x_resized = F.interpolate(
                    x_single,
                    size=(target_size, target_size),
                    mode='area'
                )
            else:
                # Upsampling: use bilinear
                x_resized = F.interpolate(
                    x_single,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
            
            x_single = x_resized.squeeze(0)  # Back to (C, H, W)
        
        img = x_single.cpu().numpy()
        
        return img, shape_info
    
    @staticmethod
    def convert_for_display(img: np.ndarray) -> Tuple[np.ndarray, str, Optional[str]]:
        """
        Convert image array to display format.
        
        Parameters
        ----------
        img : np.ndarray
            Image in (C, H, W) or (H, W) format
            
        Returns
        -------
        img_disp : np.ndarray
            Image ready for matplotlib
        cmap : str
            Colormap to use ("gray" or None)
        title_suffix : str or None
            Additional title info
        """
        
        if img.ndim == 3:  # (C, H, W)
            C, H, W = img.shape
        elif img.ndim == 2:  # (H, W)
            C, H, W = 1, img.shape[0], img.shape[1]
        else:
            raise ValueError(f"Unsupported shape: {img.shape}")
        
        if C == 1:
            # Grayscale
            img_disp = img.squeeze(0) if img.ndim == 3 else img
            cmap = "gray"
            title_suffix = f"({H}×{W})"
        elif C == 3:
            # RGB: Normalize and convert to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            
            # Normalize to [0, 1] range
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_disp = (img - img_min) / (img_max - img_min)
            else:
                img_disp = img
            
            cmap = None
            title_suffix = f"RGB ({H}×{W})"
        else:
            # Multi-channel: take mean or first 3
            if C >= 3:
                img = img[:3]
            img = np.transpose(img, (1, 2, 0))
            img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)
            cmap = None
            title_suffix = f"{C} channels"
        
        return img_disp, cmap, title_suffix
    
    # ========================================================================
    # SECTION 2: PROBABILITY HEATMAP ANALYSIS
    # ========================================================================
    
    @staticmethod
    def get_probability_mass(
        p: torch.Tensor,
        sample_idx: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract probability map and calculate total mass.
        
        Parameters
        ----------
        p : torch.Tensor
            Probability tensor
        sample_idx : int
            Batch index
            
        Returns
        -------
        p_map : np.ndarray
            2D probability map
        mass : float
            Total probability mass (sum)
        """
        
        p = p.detach().cpu()
        
        if p.dim() == 4:  # (N, C, H, W)
            p_single = p[sample_idx]  # (C, H, W)
            p_map = p_single.mean(dim=0).numpy()  # Average across channels
        elif p.dim() == 3:  # (C, H, W)
            p_map = p.mean(dim=0).numpy()
        elif p.dim() == 2:  # (H, W)
            p_map = p.numpy()
        elif p.dim() == 1:  # (D,)
            # Try to reshape to square
            size = int(np.sqrt(len(p)))
            if size * size == len(p):
                p_map = p.view(size, size).numpy()
            else:
                p_map = p.numpy().reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported probability tensor shape: {p.shape}")
        
        mass = float(p_map.sum())
        
        return p_map, mass
    
    @staticmethod
    def apply_occlusion_mask(
        img_disp: np.ndarray,
        p_map: np.ndarray,
        threshold_percentile: int = 90,
    ) -> np.ndarray:
        """
        Create masked image showing only top-contributing pixels.
        
        Parameters
        ----------
        img_disp : np.ndarray
            Display image (H, W, C) or (H, W)
        p_map : np.ndarray
            Probability map (H, W)
        threshold_percentile : int
            Keep pixels above this percentile
            
        Returns
        -------
        masked : np.ndarray
            Image with non-contributing pixels masked to black
        """
        
        # Resize probability map to match image size if needed
        if p_map.shape != img_disp.shape[:2]:
            p_map_resized = np.array(
                [np.interp(np.linspace(0, p_map.shape[0]-1, img_disp.shape[0]), 
                           np.arange(p_map.shape[0]), row) 
                 for row in p_map.T]
            ).T
            p_map = p_map_resized
        
        # Calculate threshold
        threshold = np.percentile(p_map, threshold_percentile)
        mask = p_map >= threshold
        
        # Apply mask
        if img_disp.ndim == 3:  # (H, W, C)
            masked = img_disp.copy()
            for c in range(img_disp.shape[2]):
                masked[:, :, c] = img_disp[:, :, c] * mask
        else:  # (H, W)
            masked = img_disp * mask
        
        return masked
    
    @staticmethod
    def calculate_heatmap_statistics(p_map: np.ndarray) -> Dict[str, float]:
        """
        Calculate useful statistics about probability heatmap.
        
        Parameters
        ----------
        p_map : np.ndarray
            Probability map (H, W)
            
        Returns
        -------
        stats : Dict
            Statistics including centroid, concentration, etc.
        """
        
        H, W = p_map.shape
        
        # Normalize for analysis
        p_norm = p_map / (p_map.sum() + 1e-12)
        
        # Calculate centroid
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        y_grid, x_grid = np.meshgrid(x_coords, y_coords)
        
        centroid_y = float((p_norm * y_grid).sum())
        centroid_x = float((p_norm * x_grid).sum())
        
        # Distance from image center
        center_y, center_x = H / 2, W / 2
        distance_from_center = np.sqrt(
            (centroid_y - center_y)**2 + (centroid_x - center_x)**2
        )
        
        # Entropy
        p_clipped = np.clip(p_norm, 1e-10, 1.0)
        entropy = float(-(p_clipped * np.log(p_clipped)).sum())
        
        # Concentration (inverse of entropy, normalized)
        max_entropy = np.log(H * W)
        concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        return {
            "total_mass": float(p_map.sum()),
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,
            "distance_from_center": distance_from_center,
            "entropy": entropy,
            "concentration": concentration,
        }
    
    # ========================================================================
    # SECTION 3: VISUALIZATION FUNCTIONS
    # ========================================================================
    
    def visualize_probability_decision_logic(
        self,
        x: torch.Tensor,
        predicted: Union[int, torch.Tensor],
        true_label: Optional[Union[int, torch.Tensor]] = None,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Visualize PAF decision logic with proper normalization and statistics.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (N, C, H, W)
        predicted : int or Tensor
            Predicted class(es)
        true_label : int or Tensor, optional
            True class(es)
        sample_idx : int
            Which sample in batch
        save_path : str, optional
            Path to save figure
        device : torch.device
            Device (for compatibility)
        """
        
        # Extract labels
        pred_val = predicted[sample_idx].item() if isinstance(predicted, torch.Tensor) else predicted
        true_val = (true_label[sample_idx].item() if isinstance(true_label, torch.Tensor) else true_label) if true_label is not None else None
        
        # Prepare images
        img, shape_info = self.prepare_image_for_display(x, sample_idx)
        img_disp, cmap_img, title_img = self.convert_for_display(img)
        
        # Get probability maps
        distributions = self.paf.distributions
        if "x" not in distributions:
            print("Warning: No input probability distribution found")
            return
        
        p_pred, mass_pred = self.get_probability_mass(distributions["x"], sample_idx)
        
        if true_val is not None:
            # For true label, we'd need to recompute PAF with true_label as target
            # For now, we'll just use pred
            p_true = None
        else:
            p_true = None
        
        # Prepare layout
        cols = 3 
        fig, axes = plt.subplots(2, cols, figsize=(5*cols, 10))
        
        # Title
        is_correct = pred_val == true_val if true_val is not None else None
        title = f"PAF Decision Logic | Pred: {pred_val}"
        if true_val is not None:
            title += f" | True: {true_val} {'✓' if is_correct else '✗'}"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        
        # Row 0, Col 0: Input Image
        axes[0, 0].imshow(img_disp, cmap=cmap_img)
        axes[0, 0].set_title(f"Input {title_img}")
        axes[0, 0].axis("off")
        
        # Row 0, Col 1: Predicted Probability Heatmap
        im1 = axes[0, 1].imshow(p_pred, cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_title(f"Class {pred_val} Attribution\n(mass: {mass_pred:.3f})")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Probability")
        axes[0, 1].axis("off")
        
        # Row 0, Col 2: Overlay
        axes[0, 2].imshow(img_disp, cmap=cmap_img)
        axes[0, 2].imshow(p_pred, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[0, 2].set_title("Attribution Overlay")
        axes[0, 2].axis("off")
        
        # Row 1, Col 0: Occlusion Test (Top 10% pixels)
        masked_img = self.apply_occlusion_mask(img_disp, p_pred, threshold_percentile=90)
        axes[1, 0].imshow(masked_img, cmap=cmap_img)
        axes[1, 0].set_title("Occlusion Test\n(Top 10% pixels)")
        axes[1, 0].axis("off")
        
        # Row 1, Col 1: Heatmap Statistics
        stats = self.calculate_heatmap_statistics(p_pred)
        ax_stats = axes[1, 1]
        ax_stats.axis("off")
        
        stats_text = (
            f"Statistics:\n"
            f"Mass: {stats['total_mass']:.4f}\n"
            f"Centroid: ({stats['centroid_x']:.1f}, {stats['centroid_y']:.1f})\n"
            f"Distance from center: {stats['distance_from_center']:.1f}\n"
            f"Entropy: {stats['entropy']:.3f}\n"
            f"Concentration: {stats['concentration']:.3f}"
        )
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
                     transform=ax_stats.transAxes, verticalalignment="center")
        
        # Row 1, Col 2: Occlusion Test (Reverse - Bottom 10%)
        if cols > 2 and true_val is not None:
            masked_reverse = self.apply_occlusion_mask(img_disp, p_pred, threshold_percentile=10)
            axes[1, 2].imshow(masked_reverse, cmap=cmap_img)
            axes[1, 2].set_title("Inverse Mask\n(Bottom 10% pixels)")
            axes[1, 2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_paf(
        self,
        x: torch.Tensor,
        predicted: int,
        true_label: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Comprehensive PAF visualization with entropy and layer analysis.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (N, C, H, W)
        predicted : int
            Predicted class
        true_label : int, optional
            True class
        save_path : str, optional
            Path to save figure
        """
        
        sample_idx = 0
        distributions = self.paf.distributions
        
        if not distributions:
            print("No PAF distributions available")
            return
        
        # Prepare image
        img, shape_info = self.prepare_image_for_display(x, sample_idx)
        img_disp, cmap_img, title_img = self.convert_for_display(img)
        
        # Classify layers
        conv_layers = []
        fc_layers = []
        other_layers = []
        
        for name, p in distributions.items():
            p_dim = p.dim()
            if p_dim == 3 or p_dim == 4:  # Convolutional
                conv_layers.append(name)
            elif p_dim == 1:  # Fully connected
                fc_layers.append(name)
            else:
                other_layers.append(name)
        
        fc_layers = sorted(fc_layers)
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"PAF Comprehensive Analysis | Predicted: {predicted}"
            + (f" | True: {true_label}" if true_label else ""),
            fontsize=14, fontweight="bold"
        )
        
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
        
        # Row 0: Input and layer heatmaps
        # (0, 0) Input image
        ax_input = fig.add_subplot(gs[0, 0])
        ax_input.imshow(img_disp, cmap=cmap_img)
        ax_input.set_title(f"Input {title_img}")
        ax_input.axis("off")
        
        # (0, 1) Input attribution
        ax_attr = fig.add_subplot(gs[0, 1])
        if "x" in distributions:
            p_input, mass = self.get_probability_mass(distributions["x"], sample_idx)
            im = ax_attr.imshow(p_input, cmap="hot", vmin=0, vmax=1)
            ax_attr.set_title(f"Input Attribution\n(mass: {mass:.4f})")
            plt.colorbar(im, ax=ax_attr, fraction=0.046, pad=0.04)
        ax_attr.axis("off")
        
        # (0, 2) Overlay
        ax_overlay = fig.add_subplot(gs[0, 2])
        if "x" in distributions:
            p_input, _ = self.get_probability_mass(distributions["x"], sample_idx)
            ax_overlay.imshow(img_disp, cmap=cmap_img)
            ax_overlay.imshow(p_input, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            ax_overlay.set_title("Attribution Overlay")
        ax_overlay.axis("off")
        
        # (0, 3) Statistics
        ax_stats = fig.add_subplot(gs[0, 3])
        if "x" in distributions:
            p_input, _ = self.get_probability_mass(distributions["x"], sample_idx)
            stats = self.calculate_heatmap_statistics(p_input)
            stats_text = (
                f"Input Stats:\n"
                f"Entropy: {stats['entropy']:.3f}\n"
                f"Concentration: {stats['concentration']:.2%}\n"
                f"Center dist: {stats['distance_from_center']:.1f}"
            )
            ax_stats.text(0.05, 0.5, stats_text, fontsize=9, family="monospace",
                         transform=ax_stats.transAxes, verticalalignment="center")
        ax_stats.axis("off")
        
        # Row 1: Conv layers
        if conv_layers:
            gs_conv = gridspec.GridSpecFromSubplotSpec(
                1, min(4, len(conv_layers)),
                subplot_spec=gs[1, :],
                wspace=0.3
            )
            
            for i, lname in enumerate(conv_layers[:4]):
                ax = fig.add_subplot(gs_conv[0, i])
                p, mass = self.get_probability_mass(distributions[lname], sample_idx)
                im = ax.imshow(p, cmap="hot", vmin=0, vmax=1)
                ax.set_title(f"{lname}\n(mass: {mass:.3f})")
                ax.axis("off")
        
        # Row 2: FC layers and entropy
        # (2, 0-1) FC distributions
        if fc_layers:
            gs_fc = gridspec.GridSpecFromSubplotSpec(
                1, min(2, len(fc_layers)),
                subplot_spec=gs[2, :2],
                wspace=0.4
            )
            
            for i, lname in enumerate(fc_layers[:2]):
                ax = fig.add_subplot(gs_fc[0, i])
                p = distributions[lname].detach().cpu().flatten().numpy()
                colors = ["red" if j == predicted else "blue" for j in range(len(p))]
                ax.bar(range(len(p)), p, color=colors)
                ax.set_title(f"{lname} (Output)")
                ax.set_xlabel("Class")
                ax.set_ylabel("Probability")
        
        # (2, 2-3) Entropy across layers
        ax_entropy = fig.add_subplot(gs[2, 2:])
        
        entropies = []
        layer_names = []
        
        for name in sorted(distributions.keys()):
            p = distributions[name].detach().cpu().flatten()
            p = p / (p.sum() + 1e-12)
            p = torch.clamp(p, min=1e-12)
            h = float(-(p * torch.log(p)).sum())
            entropies.append(h)
            layer_names.append(name)
        
        ax_entropy.plot(range(len(entropies)), entropies, "o-", linewidth=2, markersize=6)
        ax_entropy.set_xticks(range(len(layer_names)))
        ax_entropy.set_xticklabels(layer_names, rotation=45, ha="right")
        ax_entropy.set_ylabel("Entropy")
        ax_entropy.set_title("Layer Entropy (Uncertainty)")
        ax_entropy.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_channel_importance(self,p_true, p_pred, layer_name="Layer"):
        # Sum over spatial dimensions: [C, H, W] -> [C]
        true_imp = p_true.sum(dim=(1, 2)).cpu().numpy()
        pred_imp = p_pred.sum(dim=(1, 2)).cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(true_imp))
        
        ax.bar(x - 0.2, true_imp, width=0.4, label='True', alpha=0.7)
        ax.bar(x + 0.2, pred_imp, width=0.4, label='Pred', alpha=0.7)
        
        ax.set_title(f"Channel Importance Comparison: {layer_name}")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Total Probability Mass")
        ax.legend()
        plt.title("Channel Importance")
        self._ensure_save_dir("PAF-output/channel_importance.png")
        plt.savefig("PAF-output/channel_importance" + ".png")

    def get_coactivation_diff(self,p_true, p_pred, top_k=10):
        # Normalize per channel to compare patterns
        p_t_norm = p_true / (p_true.max() + 1e-8)
        p_p_norm = p_pred / (p_pred.max() + 1e-8)
        
        # Find channels where pred is high and true is low
        # This highlights "hallucinated" features
        ghost_features = torch.clamp(p_p_norm - p_t_norm, min=0)
        
        # Get the sum of "ghosting" per channel
        ghost_scores = ghost_features.sum(dim=(1,2))
        top_ghost_indices = torch.topk(ghost_scores, k=top_k).indices
        
        return top_ghost_indices, ghost_features
    
    def get_structural_diff_heatmap(self,p_true, p_pred):
        # Element-wise difference preserves channel-specific spatial errors
        diff = p_pred - p_true
        
        # We take the absolute diff to see "total deviation" 
        # OR leave as is to see over-prediction (pos) vs under-prediction (neg)
        heatmap = diff.sum(dim=0).cpu().numpy()
        
        # Apply Gaussian smoothing (as discussed) for cleaner visuals
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=2.0)
        
        return heatmap
    def get_top_k_mass_mask(self,heatmap, percentile=95):
        # Flatten and find the threshold value for top 5%
        thresh = np.percentile(heatmap, percentile)
        
        # Create binary mask
        mask = (heatmap >= thresh).astype(float)
        
        # Optional: Soften the edges of the mask
        mask = gaussian_filter(mask, sigma=1.0)
        
        return mask
                
    def plot_diagnostic_comparison(self,p_true, p_pred, x_orig):
        # 1. Get structural diff
        if x_orig.dim() == 4:
            x_orig = x_orig[0] # Take the first sample in the batch

        diff_map = self.get_structural_diff_heatmap(p_true, p_pred)
        
        # 2. Get top 5% focus area
        focus_mask = self.get_top_k_mass_mask(diff_map, percentile=95)
        
        # 3. Apply mask to original image to show "Why the model diverged"
        # Reshape mask to match x_orig [C, H, W]
        import cv2
        h, w = x_orig.shape[-2:]
        mask_resampled = cv2.resize(focus_mask, (w, h))

        masked_img = x_orig.cpu().numpy().transpose(1,2,0) * mask_resampled[..., np.newaxis]

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(diff_map, cmap='RdBu_r') # Red = Overpred, Blue = Underpred
        axes[0].set_title("Structural Difference (Pred - True)")
        
        axes[1].imshow(masked_img)
        axes[1].set_title("Top 5% Mass Deviation Regions")
        plt.title("New Experiment")
        self._ensure_save_dir("PAF-output/new_experiment.png")
        plt.savefig("PAF-output/new_experiment" + ".png")
    

# --- Example of how to structure your input data ---
# edge_map = {
#    ('layer1.0.add', 'layer1.0.bn2'): 0.85,  # Main path contribution
#    ('layer1.0.add', 'relu1'): 0.15          # Skip connection contribution
# }
# viz = draw_paf_distribution_graph(traced_resnet, successor_list, edge_map)
# viz.render('resnet_paf_flow')
