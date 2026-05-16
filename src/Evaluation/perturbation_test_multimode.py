from scipy import stats
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as Fn
from scipy.stats import wilcoxon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Evaluation.utils_main import get_sample_image
from core.paf.scoring import ScoringMode
from core.paf.utils import make_mode_name
from core.visualization.visualizer import PAFVisualizer
from core.paf.graph.manager import PAFGraphManager
from Evaluation.xai_integration import XAI
from Evaluation.eval_core.paf_runner import PAFRunner
from Evaluation.eval_core.baseline_runner import BaselineRunner
from Evaluation.eval_core.metrics import compute_similarity_scores, _compute_audc, _compute_auic

class PerturbationTestMultiMode:
    def __init__(
        self,
        model,
        graph_manager:             PAFGraphManager,
        model_name :                str= "",
        analyze_misclassification:  bool = False,
        contrastive_interpretation: bool = False,
        sample_idx:                 int  = 0,
        patch_size:                 int  = 1,
        visualize_heatmap:          bool = False,
        paf_modes=None,   # list of (ScoringMode, dict) — None = ABS only
    ):
        self.model                     = model
        self.xai                       = XAI(model)
        self.analyze_misclassification = analyze_misclassification
        self.contrastive_explanation   = contrastive_interpretation
        self.sample_idx                = sample_idx
        self.visualize_heatmap         = visualize_heatmap
        self.patch_size                = patch_size
        self.model_name                = model_name
        self.graph_manager             = graph_manager
        # PAF scoring modes — default to ABS for backward compatibility
        if paf_modes is None:
            self.paf_modes = [(ScoringMode.ABS, {'tau': 1.0})]
        else:
            self.paf_modes = paf_modes

        # Initialize core runners
        self._paf_runner = PAFRunner(model, graph_manager, paf_modes)
        self._baseline_runner = BaselineRunner(model, None)
        
        # Human-readable name per PAF mode e.g. 'PAF_abs_tau1.0'
        self.paf_method_names = self._paf_runner.mode_names(self.paf_modes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    
    @staticmethod
    def _auto_style(method_names):
        """
        Build style dict for any method list.
        PAF modes get red shades; baselines keep fixed colours.
        """
        base_styles = {
            'IG':       {'color': 'blue',   'marker': '^', 'ls': '-.', 'lw': 1.8},
            'GCAM':     {'color': 'gray',   'marker': 's', 'ls': '--', 'lw': 1.8},
            'LRP':      {'color': 'green',  'marker': 'd', 'ls': ':',  'lw': 1.8},
            'DEEPSHAP': {'color': 'orange', 'marker': 'v', 'ls': '--', 'lw': 1.8},
        }
        paf_names   = [n for n in method_names if n.startswith('PAF')]
        red_shades  = plt.cm.Reds(np.linspace(0.5, 0.9, max(len(paf_names), 1)))
        paf_markers = ['o', 'D', 'P', '*', 'X', 'h']

        styles = dict(base_styles)
        for i, name in enumerate(paf_names):
            styles[name] = {
                'color':  red_shades[i],
                'marker': paf_markers[i % len(paf_markers)],
                'ls':     '-',
                'lw':     2.5,
            }
        return styles

    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def _get_all_heatmaps(self, x, true_label, paf_predicted):
        """
        Returns {method_name: heatmap_np}.
        PAF: one entry per scoring mode, single forward pass.
        """
        results = {}
        H, W = x.shape[-2], x.shape[-1]

        # Baselines using core runner
        baseline_results = self._baseline_runner.run(x, true_label, H, W)
        results.update(baseline_results)

        # PAF — all modes in one pass using core runner
        try:
            with self._paf_runner:
                paf_heatmaps = self._paf_runner.get_heatmaps(
                    x, true_label, 'x', H, W, true_label
                )
                results.update(paf_heatmaps)

        except Exception as e:
            print(f"PAF failed: {e}")
            for name in self.paf_method_names:
                results[name] = None

        return results

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run_perturbation_tests(self, x, true_label, paf_predicted, device, idx=0):
        """
        Runs insertion + deletion for all methods.
        Returns (steps, results_dict) where results_dict[method] = {'del': [...], 'ins': [...]}.
        """
        heatmaps = self._get_all_heatmaps(x, true_label, paf_predicted)

        if self.visualize_heatmap:
            self.visualize_heatmaps(x, heatmaps, idx=idx)

        results     = {}
        steps_del   = None
        for name, hmap in heatmaps.items():
            if hmap is None:
                continue
            steps_del, conf_del = self.run_patch_deletion_experiment(
                x, hmap, steps=10, patch_size=self.patch_size
            )
            steps_ins, conf_ins = self.run_patch_insertion_experiment(
                x, hmap, steps=10, patch_size=self.patch_size
            )
            results[name] = {'del': conf_del, 'ins': conf_ins}

        return steps_del, results

    # ------------------------------------------------------------------
    # Insertion / Deletion
    # ------------------------------------------------------------------

    def run_patch_deletion_experiment(self, input_tensor, heatmap, steps=10, patch_size=3):
        B, C, H, W = input_tensor.shape

        with torch.no_grad():
            logits       = self.model(input_tensor)
            target_class = torch.argmax(logits, dim=1).item()
            baseline_conf = torch.softmax(logits, dim=1)[0, target_class].item()

        sorted_indices  = np.argsort(-heatmap.flatten())
        pixels_per_step = len(heatmap.flatten()) // steps

        confidences         = [baseline_conf]
        deleted_percentages = [0]
        mask = torch.zeros((1, 1, H, W), device=input_tensor.device)

        for i in range(1, steps + 1):
            start_idx = (i - 1) * pixels_per_step
            end_idx   = i * pixels_per_step
            for idx in sorted_indices[start_idx:end_idx]:
                row, col = divmod(int(idx), W)
                mask[:, :, row, col] = 1.0

            dilated_mask = Fn.max_pool2d(
                mask, kernel_size=patch_size, stride=1, padding=patch_size // 2
            ) if patch_size > 1 else mask

            working_img = input_tensor * (1 - dilated_mask)

            with torch.no_grad():
                conf = torch.softmax(
                    self.model(working_img), dim=1
                )[0, target_class].item()

            confidences.append(conf)
            deleted_percentages.append(i * (100 / steps))

        return deleted_percentages, confidences

    def run_patch_insertion_experiment(self, input_tensor, heatmap, steps=10, patch_size=3):
        B, C, H, W = input_tensor.shape

        with torch.no_grad():
            target_class = torch.argmax(
                self.model(input_tensor), dim=1
            ).item()

        sorted_indices  = np.argsort(-heatmap.flatten())
        pixels_per_step = len(heatmap.flatten()) // steps

        blurred_baseline = torchvision.transforms.functional.gaussian_blur(
            input_tensor, kernel_size=(21, 21), sigma=11.0
        )
        working_img = blurred_baseline.clone()
        mask        = torch.zeros((1, 1, H, W), device=input_tensor.device)

        confidences          = []
        inserted_percentages = []

        for i in range(0, steps + 1):
            if i > 0:
                start_idx = (i - 1) * pixels_per_step
                end_idx   = i * pixels_per_step
                for idx in sorted_indices[start_idx:end_idx]:
                    row, col = divmod(int(idx), W)
                    mask[:, :, row, col] = 1.0

                dilated_mask = Fn.max_pool2d(
                    mask, kernel_size=patch_size, stride=1, padding=patch_size // 2
                ) if patch_size > 1 else mask

                working_img = (
                    input_tensor   * dilated_mask
                    + blurred_baseline * (1 - dilated_mask)
                )

            with torch.no_grad():
                conf = torch.softmax(
                    self.model(working_img), dim=1
                )[0, target_class].item()

            confidences.append(conf)
            inserted_percentages.append(i * (100 / steps))

        return inserted_percentages, confidences

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_heatmaps(self, original_img, heatmaps_dict, alpha=0.5, cmap='jet', idx=0):
        """
        Shows original + all heatmap overlays.
        Splits into two rows when there are more than 6 methods.
        """
        if isinstance(original_img, torch.Tensor):
            if original_img.ndim == 4:
                original_img = original_img.squeeze(0)
            if original_img.shape[0] in [1, 3]:
                original_img = original_img.permute(1, 2, 0)
            img_np = original_img.detach().cpu().numpy()
        else:
            img_np = original_img
        img_np   = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        h, w     = img_np.shape[:2]
        cmap_img = 'gray' if img_np.ndim == 2 or img_np.shape[-1] == 1 else None

        valid       = {k: v for k, v in heatmaps_dict.items() if v is not None}
        items       = list(valid.items())
        max_per_row = 6
        n_methods   = len(items)

        def _render_items(axes_row, row_items, offset=0):
            for i, (name, hmap) in enumerate(row_items):
                ax = axes_row[i + offset]
                if isinstance(hmap, torch.Tensor):
                    hmap = hmap.detach().cpu().numpy()
                if hmap.shape[:2] != (h, w):
                    import cv2
                    hmap = cv2.resize(hmap, (w, h))
                hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
                ax.imshow(img_np, cmap=cmap_img)
                im = ax.imshow(hmap_norm, cmap=cmap, alpha=alpha)
                ax.set_title(name, fontsize=8)
                ax.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

        if n_methods > max_per_row:
            row1   = items[:max_per_row]
            row2   = items[max_per_row:]
            n_cols = max(max_per_row + 1, len(row2))
            fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 10))

            axes[0, 0].imshow(img_np, cmap=cmap_img)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            _render_items(axes[0], row1, offset=1)
            for j in range(len(row1) + 1, n_cols):
                axes[0, j].axis('off')

            _render_items(axes[1], row2, offset=0)
            for j in range(len(row2), n_cols):
                axes[1, j].axis('off')
        else:
            n_cols = n_methods + 1
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 5))
            if n_cols == 1:
                axes = [axes]
            axes[0].imshow(img_np, cmap=cmap_img)
            axes[0].set_title("Original")
            axes[0].axis('off')
            _render_items(axes, items, offset=1)

        plt.tight_layout()
        os.makedirs("PAF-output", exist_ok=True)
        plt.savefig(f"PAF-output/heatmap_comparison_{idx}.png", dpi=150)
        plt.close()

    def plot_faithfulness_single_test(self, steps, results, idx=0):
        """
        Single-sample insertion/deletion curves for all methods.
        """
        try:
            calc_area = np.trapezoid
        except AttributeError:
            calc_area = np.trapz

        method_names = list(results.keys())
        styles       = self._auto_style(method_names)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for name in method_names:
            s    = styles.get(name, {'color': 'black', 'marker': 'x', 'ls': '-', 'lw': 1.5})
            audc = calc_area(results[name]['del'], dx=1.0 / 100)
            auic = calc_area(results[name]['ins'], dx=1.0 / 100)
            ax1.plot(steps, results[name]['del'],
                     color=s['color'], marker=s['marker'],
                     linestyle=s['ls'], lw=s['lw'],
                     label=f"{name} (AUDC:{audc:.3f})")
            ax2.plot(steps, results[name]['ins'],
                     color=s['color'], marker=s['marker'],
                     linestyle=s['ls'], lw=s['lw'],
                     label=f"{name} (AUIC:{auic:.3f})")

        for ax, title, xlabel in [
            (ax1, r"Deletion Test (Faithfulness $\downarrow$)", "% Pixels Removed"),
            (ax2, r"Insertion Test (Sufficiency $\uparrow$)",   "% Pixels Inserted"),
        ]:
            ax.axhline(y=0.1, color='gray', linestyle=':', label='Random Chance')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Model Confidence")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        os.makedirs("PAF-output", exist_ok=True)
        plt.savefig(f"PAF-output/insertion_deletion_results_{idx}.png", dpi=300)
        plt.close()

    def plot_aggregated_results(self, steps, del_matrix, ins_matrix, method_names, audc, auic):
        """
        Aggregated mean +/- std curves for all methods.
        audc, auic: dicts {method_name: np.array of per-sample scores}
        — passed in from compute_aggregate_statistics so AUDC/AUIC
        labels match the table values exactly.
        """
        styles  = self._auto_style(method_names)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for name in method_names:
            s = styles.get(name, {
                'color': 'black', 'marker': 'x', 'ls': '-', 'lw': 1.5
            })

            # Deletion
            del_mean = del_matrix[name].mean(axis=0)
            del_std  = del_matrix[name].std(axis=0)
            # Use pre-computed AUDC from compute_aggregate_statistics
            audc_val = audc[name].mean()

            axes[0].plot(
                steps, del_mean,
                color=s['color'], marker=s['marker'],
                linestyle=s['ls'], lw=s['lw'],
                label=f"{name} (AUDC={audc_val:.3f})"
            )
            axes[0].fill_between(
                steps,
                del_mean - del_std,
                del_mean + del_std,
                alpha=0.12, color=s['color']
            )

            # Insertion
            ins_mean = ins_matrix[name].mean(axis=0)
            ins_std  = ins_matrix[name].std(axis=0)
            auic_val = auic[name].mean()

            axes[1].plot(
                steps, ins_mean,
                color=s['color'], marker=s['marker'],
                linestyle=s['ls'], lw=s['lw'],
                label=f"{name} (AUIC={auic_val:.3f})"
            )
            axes[1].fill_between(
                steps,
                ins_mean - ins_std,
                ins_mean + ins_std,
                alpha=0.12, color=s['color']
            )

        for ax, title, xlabel in zip(
            axes,
            ["Deletion — Faithfulness (lower is better)",
            "Insertion — Sufficiency (higher is better)"],
            ["% Pixels Removed", "% Pixels Inserted"],
        ):
            ax.axhline(
                y=0.1, color='black', linestyle=':', lw=1.2,
                label='Random Chance'
            )
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Mean Model Confidence", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(steps[0], steps[-1])

        plt.suptitle(
            f"Perturbation Faithfulness Test — {self.model_name}",
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        os.makedirs("PAF-output", exist_ok=True)
        plt.savefig(
            f"PAF-output/aggregated_results_{self.model_name}.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        return fig

    def plot_aggregated_results_old(self, steps, del_matrix, ins_matrix, method_names):
        """
        Aggregated mean +/- std curves for all methods.
        """
        styles = self._auto_style(method_names)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for name in method_names:
            s        = styles.get(name, {'color': 'black', 'marker': 'x', 'ls': '-', 'lw': 1.5})
            del_mean = del_matrix[name].mean(axis=0)
            del_std  = del_matrix[name].std(axis=0)
            ins_mean = ins_matrix[name].mean(axis=0)
            ins_std  = ins_matrix[name].std(axis=0)

            axes[0].plot(steps, del_mean,
                         color=s['color'], marker=s['marker'],
                         linestyle=s['ls'], lw=s['lw'],
                         label=f"{name} (AUDC={del_mean.mean():.3f})")
            axes[0].fill_between(steps, del_mean - del_std, del_mean + del_std,
                                  alpha=0.12, color=s['color'])

            axes[1].plot(steps, ins_mean,
                         color=s['color'], marker=s['marker'],
                         linestyle=s['ls'], lw=s['lw'],
                         label=f"{name} (AUIC={ins_mean.mean():.3f})")
            axes[1].fill_between(steps, ins_mean - ins_std, ins_mean + ins_std,
                                  alpha=0.12, color=s['color'])

        for ax, title, xlabel in zip(
            axes,
            ["Deletion Test - Faithfulness down", "Insertion Test - Sufficiency up"],
            ["% Pixels Removed", "% Pixels Inserted"],
        ):
            ax.axhline(y=0.1, color='black', linestyle=':', lw=1.2, label='Random Chance')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Mean Model Confidence", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        os.makedirs("PAF-output", exist_ok=True)
        plt.savefig("PAF-output/aggregated_results"+self.model_name+".png", dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_aggregate_statistics(self, all_results, method_names, steps):
        """
        AUDC/AUIC, win rates, Wilcoxon p-values, Cohen's d.
        Reference method: first PAF mode found, or 'PAF' if present.
        """
        del_matrix = {n: np.array(all_results[n]['del']) for n in method_names}
        ins_matrix = {n: np.array(all_results[n]['ins']) for n in method_names}
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

    def print_summary_table(self, stats, method_names,save_path: str = None):
        def effect_label(d):
            ad = abs(d)
            if ad >= 0.8: return "large"
            if ad >= 0.5: return "medium"
            if ad >= 0.2: return "small"
            return "negligible"

        def sig_label(p):
            if p is None:  return "  —"
            if p < 0.001:  return "***"
            if p < 0.01:   return " **"
            if p < 0.05:   return "  *"
            return "n.s."

        col_w = max(25, max(len(n) for n in method_names) + 2)
        sep   = "=" * (col_w + 100)
        lines = []

        lines.append(f"\n{sep}")
        lines.append(f"{'Method':<{col_w}} {'AUDC':>16} {'AUIC':>16} "
                    f"{'WR Del':>8} {'WR Ins':>8} "
                    f"{'p Del':>6} {'p Ins':>6} "
                    f"{'d Del':>22} {'d Ins':>22}")
        lines.append(sep)

        print(f"\n{sep}")
        print(f"{'Method':<{col_w}} {'AUDC':>16} {'AUIC':>16} "
              f"{'WR Del':>8} {'WR Ins':>8} "
              f"{'p Del':>6} {'p Ins':>6} "
              f"{'d Del':>22} {'d Ins':>22}")
        print(sep)

        ref = 'PAF' if 'PAF' in method_names else \
              next((n for n in method_names if n.startswith('PAF-')), method_names[0])

        for name in method_names:
            s   = stats[name]
            lbl = (name + " <ref>") if name == ref else name
            '''
            print(
                f"{lbl:<{col_w}} "
                f"{s['audc_mean']:.3f}+/-{s['audc_std']:.3f}    "
                f"{s['auic_mean']:.3f}+/-{s['auic_std']:.3f}  "
                f"{s['win_rate_del']:>7.1f}% "
                f"{s['win_rate_ins']:>7.1f}%  "
                f"{sig_label(s['p_del']):>6} "
                f"{sig_label(s['p_ins']):>6}  "
                + (
                    f"{s['d_del']:.3f} ({effect_label(s['d_del'])})  "
                    f"{s['d_ins']:.3f} ({effect_label(s['d_ins'])})"
                    if s['d_del'] is not None else "  —                         —"
                )
            )
            '''
            row = (
                f"{lbl:<{col_w}} "
                f"{s['audc_mean']:.3f}+/-{s['audc_std']:.3f}    "
                f"{s['auic_mean']:.3f}+/-{s['auic_std']:.3f}  "
                f"{s['win_rate_del']:>7.1f}% "
                f"{s['win_rate_ins']:>7.1f}%  "
                f"{sig_label(s['p_del']):>6} "
                f"{sig_label(s['p_ins']):>6}  "
                + (
                    f"{s['d_del']:.3f} ({effect_label(s['d_del'])})  "
                    f"{s['d_ins']:.3f} ({effect_label(s['d_ins'])})"
                    if s.get('d_del') is not None else "  —                         —"
                )
            )
            lines.append(row)
        lines.append(sep)
        lines.append(f"Reference: {ref} | *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")

        # Print to console
        print("\n".join(lines))
        #print(sep)
        #print(f"Reference: {ref} | *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
                print(f"\nSummary table saved to: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save summary to {save_path}: {e}")



# Auxiliary methods

def run_perturbation_tests(
    test_loader,
    model,
    device,
    graph_manager,
    n_samples   :int  = 100,
    model_name  :str="",
    paf_modes   :list = None,
    random_sample:bool = True,
    visualize_heatmap: bool = False,
):
    collected         = 0
    sample_id         = 0

    ptest = PerturbationTestMultiMode(
        model                      = model,
        model_name                 = model_name,
        graph_manager               = graph_manager,
        paf_modes                  = paf_modes,
    )

    # Build method names dynamically from what PerturbationTest will produce
    baseline_names = ["GradCAM++", "IG", "LRP", "DeepSHAP"]
    paf_names      = ptest.paf_method_names
    method_names   = baseline_names + paf_names

    all_results = {name: {'del': [], 'ins': []} for name in method_names}

    while collected < n_samples:
        test_sample = get_sample_image(
            test_loader, model, device,
            random_sample,
            samples_checked=sample_id
        )
        idx, x, y, predicted, sample_id = test_sample
        true_label = y.item() if isinstance(y, torch.Tensor) else y

        try:
            steps, results = ptest.run_perturbation_tests(
                x=x, true_label=true_label, paf_predicted=predicted, device=device, idx=idx
            )

            # Accumulate — only store methods that succeeded
            for name, res in results.items():
                if name in all_results:
                    for k in ('del', 'ins'):
                        all_results[name][k].append(res[k])
                else:
                    # Method appeared that was not in our initial list
                    # (should not happen but handle gracefully)
                    all_results[name] = {'del': [res['del']], 'ins': [res['ins']]}
                    method_names.append(name)
                    print(f"New method added to results: {name}")

            collected += 1
            sample_id += 1

            if visualize_heatmap:
                ptest.plot_faithfulness_single_test(steps, results, idx)

            print(f"[{collected}/{n_samples}] sample {idx} done")

        except Exception as e:
            print(f"Sample {idx} failed: {e} — skipping")
            sample_id += 1
            continue

    return steps, all_results


def plot_neurips_aggregate(steps, del_matrix, ins_matrix, audc, auic):
    """
    steps:      list of x-axis values (percentages)
    del_matrix: {method: np.array(n_samples, n_steps)}
    ins_matrix: {method: np.array(n_samples, n_steps)}
    audc:       {method: np.array(n_samples)} — pre-computed by compute_aggregate_statistics
    auic:       {method: np.array(n_samples)} — pre-computed by compute_aggregate_statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    import os

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    BASE_COLORS = {
        "IG":       "#1f77b4",
        "GradCAM++":     "#7f7f7f",
        "LRP":      "#2ca02c",
        "DeepSHAP": "#d7ff0e",
    }

    def _auto_style(method_names):
        base_styles = {
            name: {"color": color, "marker": None,
                   "ls": "--", "lw": 1, "alpha": 0.9}
            for name, color in BASE_COLORS.items()
        }
        paf_styles = {
            'abs': {
                'color': '#E31A1C', 'ls': '-',
                'lw': 1.5, 'marker': 'o', 'ms': 5
            },
            'norm': {
                'color': "#FBB299D7", 'ls': (0, (3, 1, 1, 1)),
                'lw': 1.5, 'marker': 'p', 'ms': 5
            },
            'power': {
                'color': "#FF8000E9", 'ls': (0, (5, 5)),
                'lw': 2.5, 'marker': '^', 'ms': 5
            },
            'norm_power': {
                'color': "#6A3D9AE3", 'ls': '-',
                'lw': 2.5, 'marker': 'X', 'ms': 8
            }
        }
        styles = dict(base_styles)
        for name in method_names:
            matched = False
            for key, cfg in paf_styles.items():
                if key in name:
                    styles[name] = cfg
                    matched = True
                    break
            '''  
            if not matched:
                styles[name] = {
                    'color': 'black', 'ls': '-',
                    'lw': 2.0, 'marker': 'x'
                }
                '''
        return styles

    method_names  = list(del_matrix.keys())
    styles        = _auto_style(method_names)
    x_coords      = np.array(steps)
    fig, axes     = plt.subplots(1, 2, figsize=(13, 4))
    ax_ins, ax_del = axes
    legend_storage = {"ins": [], "del": []}
    PAF_MODES = ['norm_power', 'norm', 'power', 'abs']
    def _draw_order(name):
        for i, mode in enumerate(PAF_MODES):
            if mode in name:
                return i + 100   # PAF drawn after baselines
        return 0   # baselines drawn first

    sorted_methods = sorted(method_names, key=_draw_order)
    for method in sorted_methods:
        if method not in del_matrix or method not in ins_matrix:
            print(f"Skipping {method}: Missing matrix data.")
            continue

        style  = styles.get(method, {"color": "black"})
        is_paf = any(mode in method for mode in PAF_MODES)

        # ── CHANGED: use pre-computed audc/auic instead of recomputing ──
        auc_del = audc[method].mean()   # scalar — mean over samples
        auc_ins = auic[method].mean()   # scalar — mean over samples

        for ax, matrix, auc_val, key in zip(
            [ax_ins,      ax_del],
            [ins_matrix,  del_matrix],
            [auc_ins,     auc_del],
            ["ins",       "del"],
        ):
            data = matrix[method]           # (n_samples, n_steps)
            mean = data.mean(axis=0)        # (n_steps,)
            sem  = data.std(axis=0) / np.sqrt(data.shape[0])

            ax.plot(
                x_coords, mean,
                **style,
                zorder=10 if is_paf else 5
            )

            if is_paf:
                alpha     = 0.25 if 'abs' in method else 0.06
                edgecolor = style["color"] if 'abs' in method else 'none'
                zorder    = 12 if 'abs' in method else 7

                ax.fill_between(
                    x_coords,
                    mean - sem,
                    mean + sem,
                    color=style["color"],
                    alpha=alpha,
                    edgecolor=edgecolor,
                    linewidth=0.5,
                    zorder=zorder,
                )

            # ── CHANGED: label uses pre-computed AUC ──
            label  = f"{method} ({auc_val:.3f})"
            handle = Line2D([0], [0], **style, label=label)
            legend_storage[key].append((auc_val, handle))

    for ax, title, xlabel, key in zip(
        [ax_ins,                        ax_del],
        ["Insertion: Confidence ↑",     "Deletion: Confidence ↓"],
        ["% pixels inserted",           "% pixels deleted"],
        ["ins",                         "del"],
    ):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Confidence")
        ax.set_xlim(x_coords[0], x_coords[-1])
        ax.set_xticks(np.linspace(x_coords[0], x_coords[-1], 3))
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend_storage[key].sort(
            key=lambda x: x[0],
            reverse=(key == "ins")
        )
        ax.legend(
            handles=[h for _, h in legend_storage[key]],
            loc="lower right" if key == "ins" else "upper right",
            frameon=True, framealpha=0.5, fontsize=6,
        )

    plt.tight_layout()
    os.makedirs("PAF-output", exist_ok=True)
    plt.savefig("PAF-output/NeurIPS_Faithfulness.pdf", bbox_inches="tight")
    plt.savefig("PAF-output/NeurIPS_Faithfulness.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_neurips_aggregate_old(steps, global_metrics):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    import os

    # --- NeurIPS-style config ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --- Colorblind-safe palette ---
    BASE_COLORS = {
        "IG": "#1f77b4",       # blue
        "GCAM": "#7f7f7f",     # gray
        "LRP": "#2ca02c",      # green
        "DEEPSHAP": "#ff7f0e", # orange
    }

    def _auto_style(method_names):
        base_styles = {}

        # baseline methods
        for name, color in BASE_COLORS.items():
            base_styles[name] = {
                "color": color,
                "marker": None,
                "ls": "--",
                "lw": 1,
                "alpha": 0.9
            }
        paf_styles = {
            'PAF-abs': {
                'color': '#E31A1C', # Saturated Red
                'ls': '-',          # Solid
                'lw': 1.5, 
                'marker': 'o', 
                'ms': 5
            },
            'PAF-norm': {
                'color': '#FB9A99', # Soft Coral/Pink-Red
                'ls': (0, (3, 1, 1, 1)), # Dash-dot-dot
                'lw': 1.5, 
                'marker': 'p', 
                'ms': 5
            },
            'PAF-power': {
                'color': '#FF7F00', # Vibrant Orange (Taken back from DeepSHAP)
                'ls': (0, (5, 5)),  # Long Dash
                'lw': 2.5, 
                'marker': '^', 
                'ms': 5
            },
            'PAF-norm_power': {
                'color': '#6A3D9A', # Deep Royal Purple (The "Premium" Variant)
                'ls': '-',          # Solid
                'lw': 2.5,          # Thickest to stand out
                'marker': 'X',      # Star marker
                'ms': 8             # Larger marker
            }
        }
        styles = dict(base_styles)
        # PAF methods (highlighted)
        paf_names = sorted([n for n in method_names if n.startswith("PAF-")])
        reds = plt.cm.Reds(np.linspace(0.5, 0.9, max(len(paf_names), 1)))

        for name in paf_names:
            matched = False
            # Match by key (e.g., 'PAF_norm_power' matches 'PAF_norm_power_tau2.0')
            for key, cfg in paf_styles.items():
                if key in name:
                    styles[name] = cfg
                    matched = True
                    break
            
            # Fallback for unexpected naming
            if not matched:
                styles[name] = {'color': 'black', 'ls': '-', 'lw': 2.0, 'marker': 'x'}

        return styles

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_ins, ax_del = axes

    legend_storage = {"ins": [], "del": []}
    styles = _auto_style(list(global_metrics.keys()))
    x_coords = np.array(steps)

    trap_fn = getattr(np, "trapezoid", np.trapezoid)

    for method, data in global_metrics.items():
        if data is None or not data.get("ins") or not data.get("del"):
            print(f"Skipping {method}: Missing data.")
            continue
        try:
            style = styles.get(method, {"color": "black"})
            is_paf = method.startswith("PAF")

            # Filter out any None or empty entries within the lists
            valid_ins = [np.atleast_2d(run) for run in data["ins"] if run is not None and np.array(run).size > 0]
            valid_del = [np.atleast_2d(run) for run in data["del"] if run is not None and np.array(run).size > 0]
            
            if not valid_ins or not valid_del:
                print(f"Skipping {method}: No valid arrays to concatenate.")
                continue

            all_ins = np.concatenate(valid_ins, axis=0)
            all_del = np.concatenate(valid_del, axis=0)
        except ValueError as e:
            print(f"Skipping {method} due to dimension mismatch: {e}")
            continue
        for ax, curve_data, key in zip(
            [ax_ins, ax_del],
            [all_ins, all_del],
            ["ins", "del"]
        ):
            mean = np.mean(curve_data, axis=0)
            sem  = np.std(curve_data, axis=0) / np.sqrt(curve_data.shape[0])

            auc = trap_fn(mean, x_coords) / (x_coords[-1] - x_coords[0])

            ax.plot(
                x_coords,
                mean,
                **style,
                zorder=10 if is_paf else 5
            )

            if is_paf:
                ax.fill_between(
                    x_coords,
                    mean - sem,
                    mean + sem,
                    color=style["color"],
                    alpha=0.15,
                    edgecolor=style["color"],
                    linewidth=0.5,
                )

            label = f"{method} ({auc:.3f})"
            handle = Line2D([0], [0], **style, label=label)
            legend_storage[key].append((auc, handle))

    # --- Formatting ---
    for ax, title, key in zip(
        [ax_ins, ax_del],
        ["Insertion: Confidence ↑", "Deletion: Confidence ↓"],
        ["ins", "del"]
    ):
        ax.set_title(title)
        if key=="ins":
            ax.set_xlabel("% pixels inserted")
        else:
            ax.set_xlabel("% pixels deleted")

        ax.set_ylabel("Confidence")

        x_min, x_max = x_coords[0], x_coords[-1]
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.linspace(x_min, x_max, 3))
        ax.set_xticklabels(["0%", "50%", "100%"])

        ax.set_ylim(-0.02, 1.05)

        # subtle grid
        ax.grid(True, linestyle=":", alpha=0.3)

        # remove clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --- Legend sorted by AUC ---
        legend_storage[key].sort(
            key=lambda x: x[0],
            reverse=(key == "ins")
        )
        handles = [h for _, h in legend_storage[key]]

        ax.legend(
            handles=handles,
            loc="lower right" if key == "ins" else "upper right",
            frameon=True,
            framealpha=0.5,
            fontsize=6,
        )

    # --- Save properly ---
    os.makedirs("PAF-output", exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        "PAF-output/NeurIPS_Faithfulness.pdf",
        bbox_inches="tight"
    )
    plt.savefig(
        "PAF-output/NeurIPS_Faithfulness.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ----------------------------------------------------------------
    # Legends — separate per subplot, sorted by AUC
    # Insertion: higher AUC is better → sort descending
    # Deletion:  lower AUC is better  → sort ascending
    # ----------------------------------------------------------------
    def sort_handles(handles, ascending=False):
        """Sort legend handles by AUC value embedded in label."""
        def extract_auc(h):
            try:
                return float(h.get_label().split("AUC=")[1].rstrip("]"))
            except Exception:
                return 0.0
        return sorted(handles, key=extract_auc, reverse=not ascending)
def generate_summary_table(
    stats,           # from compute_aggregate_statistics
    audc,            # from compute_aggregate_statistics
    auic,            # from compute_aggregate_statistics
    method_names,    # list of method names in order
    output_dir="PAF-output",
    filename="summary_table.txt"
):
    table_lines = []

    def smart_print(text):
        print(text)
        table_lines.append(text)

    def effect_label(d):
        ad = abs(d)
        if ad >= 0.8: return "large"
        if ad >= 0.5: return "medium"
        if ad >= 0.2: return "small"
        return "negligible"

    def sig_label(p):
        if p is None:  return "  —"
        if p < 0.001:  return "***"
        if p < 0.01:   return " **"
        if p < 0.05:   return "  *"
        return "n.s."

    # ── REMOVED: AUC recomputation block ──
    # ── CHANGED: use pre-computed audc/auic directly ──

    # Identify reference method — same logic as compute_aggregate_statistics
    ref = next(
        (n for n in ["DEEPSHAP", "GCAM"] if n in method_names),
        method_names[0]
    )

    col_w = max(25, max(len(n) for n in method_names) + 2)
    sep   = "=" * (col_w + 100)

    smart_print(f"\n{sep}")
    smart_print(
        f"{'Method':<{col_w}} {'AUDC (Del) ↓':>18} {'AUIC (Ins) ↑':>18} "
        f"{'p Del':>6} {'p Ins':>6} "
        f"{'d Del':>22} {'d Ins':>22}"
    )
    smart_print(sep)

    for name in method_names:
        # ── CHANGED: read directly from pre-computed stats/audc/auic ──
        s         = stats[name]
        audc_mean = s['audc_mean']
        audc_std  = s['audc_std']
        auic_mean = s['auic_mean']
        auic_std  = s['auic_std']
        p_del     = s['p_del']
        p_ins     = s['p_ins']
        d_del     = s['d_del']
        d_ins     = s['d_ins']

        lbl = (name + " [ref]") if name == ref else name
        row = (
            f"{lbl:<{col_w}} "
            f"{audc_mean:.3f}+/-{audc_std:.3f}    "
            f"{auic_mean:.3f}+/-{auic_std:.3f}  "
            f"{sig_label(p_del):>6} "
            f"{sig_label(p_ins):>6}  "
        )

        if d_del is not None:
            row += (
                f"{d_del:7.3f} ({effect_label(d_del):>10})  "
                f"{d_ins:7.3f} ({effect_label(d_ins):>10})"
            )
        else:
            row += "  —                         —"
        smart_print(row)

    smart_print(sep)
    smart_print(
        f"Reference Baseline: {ref} | Wilcoxon Rank-Sums test | "
        f"*** p<0.001 ** p<0.01 * p<0.05"
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))

def generate_summary_table_old(global_metrics, output_dir="PAF-output", filename="summary_table.txt"):
    """
    Aggregates results across models/runs, prints a NeurIPS-ready table, 
    and saves the exact output to a text file.
    """
    table_lines = []

    def smart_print(text):
        print(text)
        table_lines.append(text)

    def effect_label(d):
        ad = abs(d)
        if ad >= 0.8: return "large"
        if ad >= 0.5: return "medium"
        if ad >= 0.2: return "small"
        return "negligible"

    def sig_label(p):
        if p is None:  return "  —"
        if p < 0.001:  return "***"
        if p < 0.01:   return " **"
        if p < 0.05:   return "  *"
        return "n.s."

    # Robust fallback for trapezoid (fixes NumPy AttributeError)
    trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if trap_fn is None:
        def trap_fn(y, x): return np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0)
    
    # 1. Pre-calculate AUCs locally [0, 1] per run
    method_results = {}
    for method, data in global_metrics.items():
        ins_aucs = [trap_fn(run.flatten(), np.linspace(0, 1, len(run.flatten()))) for run in data['ins']]
        del_aucs = [trap_fn(run.flatten(), np.linspace(0, 1, len(run.flatten()))) for run in data['del']]
        
        method_results[method] = {
            'ins': np.array(ins_aucs),
            'del': np.array(del_aucs)
        }

    # 2. Identify Reference Method
    method_names = list(global_metrics.keys())
    ref = next((n for n in ["DEEPSHAP", "GCAM"] if n in method_names), method_names[0])

    # 3. Table Formatting
    col_w = max(25, max(len(n) for n in method_names) + 2)
    sep = "=" * (col_w + 100)
    
    smart_print(f"\n{sep}")
    smart_print(f"{'Method':<{col_w}} {'AUDC (Del) ↓':>18} {'AUIC (Ins) ↑':>18} "
                f"{'p Del':>6} {'p Ins':>6} "
                f"{'d Del':>22} {'d Ins':>22}")
    smart_print(sep)

    # 4. Calculate Stats & Compare to Ref
    ref_ins = method_results[ref]['ins']
    ref_del = method_results[ref]['del']

    for name in method_names:
        m_ins, m_del = method_results[name]['ins'], method_results[name]['del']
        
        audc_mean, audc_std = np.mean(m_del), np.std(m_del)
        auic_mean, auic_std = np.mean(m_ins), np.std(m_ins)

        if name != ref and len(m_ins) > 1:
            _, p_del = stats.ranksums(m_del, ref_del)
            _, p_ins = stats.ranksums(m_ins, ref_ins)
            
            def cohen_d(x, y):
                nx, ny = len(x), len(y)
                std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
                pooled_std = np.sqrt(((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / (nx + ny - 2))
                return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-9)

            d_del = cohen_d(m_del, ref_del)
            d_ins = cohen_d(m_ins, ref_ins)
        else:
            p_del = p_ins = d_del = d_ins = None

        lbl = (name + " [ref]") if name == ref else name
        row = (
            f"{lbl:<{col_w}} "
            f"{audc_mean:.3f}+/-{audc_std:.3f}    "
            f"{auic_mean:.3f}+/-{auic_std:.3f}  "
            f"{sig_label(p_del):>6} "
            f"{sig_label(p_ins):>6}  "
        )
        
        if d_del is not None:
            row += f"{d_del:7.3f} ({effect_label(d_del):>10})  {d_ins:7.3f} ({effect_label(d_ins):>10})"
        else:
            row += "  —                         —"
        smart_print(row)

    smart_print(sep)
    smart_print(f"Reference Baseline: {ref} | Wilcoxon Rank-Sums test | *** p<0.001 ** p<0.01 * p<0.05")

    # 5. Save to File
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
