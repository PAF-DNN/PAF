import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import DeepLiftShap, IntegratedGradients
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import copy
import os
import tracemalloc
import time

from Evaluation.utils_main import get_sample_image
from Evaluation.xai_integration import XAI
from core.visualization.visualizer import PAFVisualizer
from core.paf.scoring import ScoringMode
from Evaluation.eval_core.paf_runner import PAFRunner
from Evaluation.eval_core.baseline_runner import BaselineRunner
from Evaluation.eval_core.metrics import compute_similarity_scores, spearman_correlation, ssim_score
from torchvision.models import ResNet18_Weights


class RandomizationTestMultiMode:
    def __init__(
        self,
        model,
        device,
        graph_manager,
        analyze_misclassification=False,
        contrastive_interpretation=False,
        sample_idx=0,
        patch_size=1,
        visualize_heatmap=False,
        paf_modes=None,   # list of (ScoringMode, dict) — if None uses ABS only
        model_name="",
    ):
        self.model = model
        self.device = device
        self.xai = XAI(model)
        self.analyze_misclassification = analyze_misclassification
        self.contrastive_explanation = contrastive_interpretation
        self.sample_idx = sample_idx
        self.visualize_heatmap = visualize_heatmap
        self.patch_size = patch_size
        self.stages = ["original", "last_conv", "full"]
        self.stage_titles = ["Trained", "Random-last conv", "Random-Full"]
        self.model_name=model_name
        self.graph_manager=graph_manager
        # PAF scoring modes — default to ABS only for backward compatibility
        if paf_modes is None:
            self.paf_modes = [(ScoringMode.ABS, {'tau': 1.0})]
        else:
            self.paf_modes = paf_modes

        # Initialize core runners
        self._paf_runner = PAFRunner(model, graph_manager, paf_modes)
       
        

    def cleanup(self):
        if hasattr(self, 'xai') and self.xai is not None:
            try:
                self.xai.cleanup()          # ← Add parentheses
            except:
                pass
            self.xai = None

        # Clear any remaining PAF hooks if you have direct access
        if hasattr(self, 'model') and self.model is not None:
            for module in self.model.modules():
                if hasattr(module, '_forward_hooks'):
                    module._forward_hooks.clear()
                if hasattr(module, '_forward_pre_hooks'):
                    module._forward_pre_hooks.clear()

        torch.cuda.empty_cache()     

    # ------------------------------------------------------------------
    # Weight randomisation
    # ------------------------------------------------------------------

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
            m.reset_running_stats()

    def get_randomized_model_at_stage(self, stage="original"):
        model_copy = copy.deepcopy(self.model)

        if stage == "original":
            return model_copy.eval()

        elif stage == "last_conv":
            # Find all layers from the last conv onwards and randomise them
            # This includes: last conv, any subsequent conv, BN, FC, classifier
            all_modules = list(model_copy.named_modules())

            # Find index of the last Conv2d
            last_conv_idx = None
            for i, (name, module) in enumerate(all_modules):
                if isinstance(module, nn.Conv2d):
                    last_conv_idx = i

            if last_conv_idx is None:
                print("Warning: No Conv2d found.")
                return model_copy.eval()

            # Randomise all modules from last_conv_idx onwards
            randomised = []
            for i, (name, module) in enumerate(all_modules):
                if i < last_conv_idx:
                    continue  # keep early layers intact
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(module.weight, mean=0, std=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                    randomised.append(name)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                    module.reset_running_stats()
                    randomised.append(name)

            print(f"Randomised {len(randomised)} layers from last conv onwards: "
                f"{randomised[:3]}{'...' if len(randomised) > 3 else ''}")
            return model_copy.eval()

        elif stage == "full":
            model_copy.apply(self.weights_init)
            return model_copy.eval()
    
    def get_randomized_model(self):
        model_rand = copy.deepcopy(self.model)
        model_rand.apply(self.weights_init)
        model_rand.eval()
        return model_rand.to(self.device)

    def get_perturbed_input(self, x: torch.Tensor, stage: str) -> torch.Tensor:
        """
        Returns a perturbed version of x for activation randomisation.
        Mirrors the progressive structure of weight randomisation:
        'original'   — unchanged input
        'last_conv'  — moderate noise (sigma = x.std())
                        analogous to randomising last conv onwards
        'full'       — pure Gaussian noise
                        analogous to full weight randomisation
        """
        if stage == "original":
            return x.clone()

        elif stage == "last_conv":
            # Moderate noise — same magnitude as signal
            # Significantly changes activations but preserves rough structure
            sigma = x.std().item()
            noise = torch.randn_like(x) * sigma
            return (x + noise).clamp(x.min().item(), x.max().item())

        elif stage == "full":
            # Pure noise — completely different activations
            # Same mean and std as original to keep dynamic range comparable
            return torch.randn_like(x) * x.std().item() + x.mean().item()

    def run_activation_randomization(
        self,
        input_x:    torch.Tensor,
        target_y:   int,
        predicted:  int,
        sample_id:  int = 0,
    ) -> dict:
        """
        Activation randomisation sanity check.
        Keeps model fixed, perturbs input across stages.
        Computes Spearman and SSIM of each method's heatmap
        vs its heatmap on the original input.

        A faithful method should score LOW — its heatmap should
        change when activations change.
        """
        input_x = input_x.to(self.device)
        stage_results = {}

        for stage in self.stages:
            #print(f"Activation randomisation stage: {stage}...")

            # Perturb input — model stays the same
            x_perturbed = self.get_perturbed_input(input_x, stage)

            # Fresh XAI instance — same model, different input
            self.xai = XAI(self.model)

            stage_results[stage] = self.run_all_methods(
                self.model, x_perturbed, target_y, self.device, predicted
            )

            #produce heatmaps of stages
            '''
            self.visualize_comparison(
                x_perturbed,
                stage_results[stage],
                title_prefix=f"ActRand {stage} — ",
                save_path=(
                    f"PAF-output/act_randomization_{stage}_{sample_id}.png"
                )
            )
            '''

        # Compute similarity vs original
        #print("\n=== Activation Randomisation Results ===")
        #print(f"{'Method':<25} | {'Stage':<12} | {'Spearman':>10} | {'SSIM':>10}")
        #print("-" * 65)

        comparison_scores = {
            name: {} for name in stage_results['original'].keys()
        }

        for name in stage_results['original']:
            h_original = stage_results['original'].get(name)
            for stage in [s for s in self.stages if s != "original"]:
                h_perturbed = stage_results[stage].get(name)
                spear    = spearman_correlation(h_original, h_perturbed)
                ssim_val = ssim_score(h_original, h_perturbed)
                comparison_scores[name][stage] = {
                    'spearman': spear,
                    'ssim':     ssim_val,
                }
                spear_str = f"{spear:.3f}" if spear is not None else "N/A"
                ssim_str  = f"{ssim_val:.3f}" if ssim_val is not None else "N/A"
                #print(f"{name:<25} | {stage:<12} | {spear_str:>10} | {ssim_str:>10}")

        self._plot_multi_stage_scores(
            comparison_scores,
            save_path=f"PAF-output/act_randomization_summary_{self.model_name}.png"
        )

        return comparison_scores
    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def run_all_methods(self, model, input_x, target_y, device, paf_predicted):
        """
        Returns dict {method_name: heatmap_np}.
        PAF produces one entry per scoring mode.
        """
        results = {}
        memory_dict={}
        time_dict={}
        H, W = input_x.shape[-2], input_x.shape[-1]
        

        # ── Update graph_manager to use current model (randomised or original) ──
        self.graph_manager._model = model
        self.graph_manager._activation_store._interpreter.module = \
            self.graph_manager._graph_info.traced
        # Update traced graph weights to match current model
        self.graph_manager._graph_info.traced.load_state_dict(
            model.state_dict(), strict=False
        )
        paf_runner      = PAFRunner(model, self.graph_manager, self.paf_modes)
        paf_method_names = paf_runner.mode_names(self.paf_modes)
        
        baseline_runner = BaselineRunner(model, None)

        # --- Baseline methods using core runner ---
        tracemalloc.start()
        start_time = time.perf_counter()
        baseline_results = baseline_runner.run(input_x, target_y, H, W)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        duration = end_time - start_time
        peak_mb = peak / (1024 * 1024)
        
        for method_name, heatmap in baseline_results.items():
            results[method_name] = heatmap
            time_dict[method_name] = duration
            memory_dict[method_name] = peak_mb
            print("Method: {}, Time: {:.3f}, Memory: {:.3f}".format(method_name, duration, peak_mb))

        # --- PAF — all modes in one forward pass using core runner ---
        try:
            tracemalloc.start()
            start_time = time.perf_counter()
            
            paf_heatmaps = paf_runner.get_heatmaps(
                input_x, target_y, 'conv1', H, W, target_y
            )
            
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            duration = end_time - start_time
            peak_mb = peak / (1024 * 1024)
            
            # Add PAF results with timing/memory info
            for method_name, heatmap in paf_heatmaps.items():
                results[method_name] = heatmap
                time_dict[method_name] = duration
                memory_dict[method_name] = peak_mb
                print("Method: {}, Time: {:.3f}, Memory: {:.3f}".format(method_name, duration, peak_mb))
                
        except Exception as e:
            print(f"PAF failed: {e}")
            for method_name in paf_method_names:
                results[method_name] = None
        finally:
            paf_runner.cleanup()

        return results, time_dict, memory_dict

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def save_run_to_csv(self, sample_id, time_dict, memory_dict, filename="PAF-output/experiment_log.csv"):
        rows = []
        
        # Iterate through the stages (e.g., 'last_conv', 'classifier.0')
        for stage in time_dict:
            # Iterate through the methods (e.g., 'PAF', 'DeepSHAP')
            for method in time_dict[stage]:
                # Get the values, defaulting to 0 if missing
                d = time_dict[stage].get(method, 0.0)
                m = memory_dict.get(stage, {}).get(method, 0.0)
                
                # Create a single row for this specific run
                rows.append({
                    'sample_id': str(sample_id),
                    'stage': stage,
                    'method': method,
                    'duration_sec': float(d),
                    'peak_memory_mb': float(m)
                })
        
        if rows:
            df = pd.DataFrame(rows)
            # Write header only if file doesn't exist
            file_exists = os.path.isfile(filename)
            df.to_csv(filename, mode='a', index=False, header=not file_exists)

    def show_final_accumulation(self, filename="PAF-output/experiment_log.csv"):
        if not os.path.exists(filename):
            print("No results file found.")
            return

        df = pd.read_csv(filename)
        
        # Calculate key metrics for the paper
        # Duration: Mean and Std (shows consistency)
        # Memory: Mean and Max (Max shows the OOM threshold)
        summary = df.groupby(['method', 'stage']).agg({
            'duration_sec': ['mean', 'std'],
            'peak_memory_mb': ['mean', 'max']
        }).round(3)
        
        print(f"\n--- Final Accumulated Results (N={df['sample_id'].nunique()} samples) ---")
        print(summary)
        
        # Optional: Get a quick total time spent per method
        total_time = df.groupby('method')['duration_sec'].sum().sort_values()
        print("\n--- Total Computation Time (Sum) ---")
        print(total_time)
        
        return summary

    def run_single(
        self,
        input_x:   torch.Tensor,
        target_y:  int,
        device,
        predicted: int,
        sample_id: int  = 0,
        test_type: str  = 'weight',   # 'weight' or 'activation'
    ) -> dict:
        """
        Progressive sanity check — weight or activation randomisation.

        Weight:     model changes, input fixed
        Activation: input changes, model fixed
        """
        input_x = input_x.to(self.device)
        stage_results = {}
        time_per_run={}
        memory_per_run={}
        for stage in self.stages:
            print(f"[{test_type}] Stage: {stage}...")

            if test_type == 'weight':
                # Model changes, input fixed
                current_model = self.get_randomized_model_at_stage(stage)
                current_model.to(self.device)
                self.xai = XAI(current_model)
                #self.xai.cleanup
                #del self.xai
                x_input = input_x

            elif test_type == 'activation':
                # Input changes, model fixed
                current_model = self.model
                self.xai      = XAI(current_model)
                x_input       = self.get_perturbed_input(input_x, stage)
                #self.xai.cleanup
                #del self.xai
            else:
                raise ValueError(f"Unknown test_type: {test_type}. Use 'weight' or 'activation'.")

            stage_results[stage], time_per_run[stage], memory_per_run[stage] = self.run_all_methods(
                current_model, x_input, target_y, device, predicted
            )

            # uncomment if needs to see heatmap
            '''
            self.visualize_comparison(
                x_input,
                stage_results[stage],
                title_prefix=f"{stage.capitalize()} — ",
                save_path=(
                    f"PAF-output/{test_type}_randomization_{stage}_{sample_id}.png"
                )
            )
            '''

            if test_type == 'weight' and stage != 'original':
                del current_model
                torch.cuda.empty_cache()

        # Compute similarity vs original
        test_label = "Weight" if test_type == 'weight' else "Activation"
        #print(f"\n=== {test_label} Randomisation Results ===")
        #print(f"{'Method':<25} | {'Stage':<12} | {'Spearman':>10} | {'SSIM':>10}")
        #print("-" * 65)

        comparison_scores = {
            name: {} for name in stage_results['original'].keys()
        }

        for name in stage_results['original']:
            h_original = stage_results['original'].get(name)
            for stage in [s for s in self.stages if s != 'original']:
                h_perturbed = stage_results[stage].get(name)
                spear    = spearman_correlation(h_original, h_perturbed)
                ssim_val = ssim_score(h_original, h_perturbed)
                comparison_scores[name][stage] = {
                    'spearman': spear,
                    'ssim':     ssim_val,
                }
                spear_str = f"{spear:.3f}" if spear is not None else "N/A"
                ssim_str  = f"{ssim_val:.3f}" if ssim_val is not None else "N/A"
               # print(f"{name:<25} | {stage:<12} | {spear_str:>10} | {ssim_str:>10}")

        '''
        self._plot_multi_stage_scores(
            comparison_scores,
            save_path=(
                f"PAF-output/{test_type}_randomization_summary_{sample_id}.png"
            )
        )
        '''
        self.save_run_to_csv(sample_id=sample_id,time_dict=time_per_run,memory_dict=memory_per_run)
        return comparison_scores


    def run(
        self,
        all_samples,
        labels,
        save_path: str = None,
        test_type: str = 'weight',    # 'weight', 'activation', or 'both'
    ) -> None:
        """
        Runs sanity check over all samples.
        test_type='both' runs weight and activation randomisation
        and saves separate aggregated plots for each.
        """
        counter   = 0
        test_runs = len(all_samples)

        # One results list per test type
        test_types = ['weight', 'activation'] if test_type == 'both' else [test_type]
        results    = {tt: [] for tt in test_types}

        for sample in all_samples:
            idx, x, y, predicted, sample_id = sample.values()
            try:
                print(f"Sample {counter}, Label: {labels[y]}")

                for tt in test_types:
                    scores = self.run_single(
                        x, y, self.device, predicted, sample_id,
                        test_type=tt
                    )
                    counter_tt = counter  # same sample, different test

                    for method, stages in scores.items():
                        for stage, metrics in stages.items():
                            results[tt].append({
                                'sample_idx': idx,
                                'true_label': y,
                                'predicted':  predicted,
                                'method':     method,
                                'stage':      stage,
                                'ssim':       metrics['ssim'],
                                'spearman':   abs(metrics['spearman'])
                                            if metrics['spearman'] is not None
                                            else None,
                                'is_paf':     method.startswith('PAF_'),
                            })

                counter += 1
                print(f"Sample {counter}/{test_runs} done")

            except Exception as e:
                print(f"Sample {idx} failed: {e}, skipping")
                continue

        # Print and save aggregated results per test type
        for tt in test_types:
            if not results[tt]:
                continue
            df = pd.DataFrame(results[tt])
            base = os.path.splitext(save_path)[0] if save_path else \
                f"PAF-output/aggregated_{tt}_randomization"
            print(f"\n=== Aggregated {tt.capitalize()} Randomisation ===")
            self.print_final_scores(df, counter, save_path=f"{base}_{tt}.png")
            # 1. Save the numerical data to CSV (This was missing)
            df.to_csv(f"{base}_{tt}.csv", index=False)
            print(f"Saved numerical results to: {base}.csv")
        
        self.show_final_accumulation()
        return results
    # ------------------------------------------------------------------
    # Plotting — updated for multi-mode
    # ------------------------------------------------------------------

    def _plot_multi_stage_scores(self, comparison_scores, save_path=None):
        """
        Plots progressive randomization scores.
        If there are many methods (e.g. multiple PAF modes), uses two rows.
        """
        methods = list(comparison_scores.keys())
        if not methods:
            return

        stages = list(comparison_scores[methods[0]].keys())
        if 'full' in stages:
            stages.remove('full')
            stages.append('full')

        n_methods = len(methods)
        n_stages  = len(stages)
        n_metrics = 2  # Spearman + SSIM
        total_bars = n_stages * n_metrics
        width = 0.8 / total_bars

        # Split into two rows if more than 6 methods
        max_per_row = 6
        if n_methods > max_per_row:
            rows   = 2
            splits = [
                methods[:n_methods // 2],
                methods[n_methods // 2:]
            ]
        else:
            rows   = 1
            splits = [methods]

        colors = plt.cm.tab20(np.linspace(0, 1, total_bars))
        fig, axes = plt.subplots(rows, 1, figsize=(max(14, max_per_row * 2), 6 * rows))

        if rows == 1:
            axes = [axes]

        for row_idx, row_methods in enumerate(splits):
            ax = axes[row_idx]
            x  = np.arange(len(row_methods))
            offset = -(total_bars - 1) * width / 2
            color_idx = 0

            for stage in stages:
                spears = [
                    abs(comparison_scores[m][stage].get('spearman') or 0)
                    for m in row_methods
                ]
                ssims = [
                    comparison_scores[m][stage].get('ssim') or 0
                    for m in row_methods
                ]

                rects_sp = ax.bar(
                    x + offset, spears, width,
                    label=f'|Spearman| ({stage})',
                    color=colors[color_idx]
                )
                self._autolabel(ax, rects_sp)
                offset    += width
                color_idx += 1

                rects_ss = ax.bar(
                    x + offset, ssims, width,
                    label=f'SSIM ({stage})',
                    color=colors[color_idx]
                )
                self._autolabel(ax, rects_ss)
                offset    += width
                color_idx += 1

            ax.axhline(
                y=0.3, color='red', linestyle='--',
                linewidth=1.5, label='Pass threshold (0.3)'
            )

            # Highlight PAF bars with a subtle background
            for i, m in enumerate(row_methods):
                if m.startswith('PAF_'):
                    ax.axvspan(
                        i - 0.45, i + 0.45,
                        alpha=0.06, color='blue',
                        label='PAF mode' if i == 0 else None
                    )

            ax.set_ylabel('Similarity to Original', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(
                row_methods, fontsize=9, fontweight='bold', rotation=15, ha='right'
            )
            ax.set_ylim(0, 1.25)
            ax.grid(True, alpha=0.3, axis='y')

            if row_idx == 0:
                ax.legend(
                    loc='upper left',
                    bbox_to_anchor=(1, 1),
                    fontsize=8
                )

        fig.suptitle(
            'Progressive Randomisation Sanity Check\n'
            'Lower scores = Better Model Fidelity',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        final_path = save_path or "PAF-output/progressive_randomization.png"
        os.makedirs("PAF-output", exist_ok=True)
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def print_final_scores(self, df_results, counter, save_path=None):
        os.makedirs("PAF-output", exist_ok=True)
        base = os.path.splitext(save_path)[0] if save_path else "PAF-output/aggregated_randomization"

        for metric in ['ssim', 'spearman']:
            df_metric = df_results[['method', 'stage', metric]].copy()
            df_metric = df_metric.rename(columns={metric: 'Score'})

            n_methods = df_results['method'].nunique()
            aspect    = max(1.0, n_methods / 5)

            g = sns.catplot(
                data      = df_metric,
                x         = 'method',
                y         = 'Score',
                hue       = 'stage',
                kind      = 'box',
                palette   = "Set2",
                height    = 6,
                aspect    = aspect,
                showmeans = True,
                meanprops = {
                    "marker":          "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize":      "6",
                }
            )

            for ax in g.axes.flat:
                ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.6, label='Threshold (0.3)')
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', labelrotation=20, labelsize=8)
                plt.setp(ax.get_xticklabels(), ha='right')

            g.set_axis_labels("Method", "Similarity Score")
            g.set_titles(f"{metric.upper()} Analysis")
            g.fig.suptitle(
                f"{metric.upper()} Sanity Check (N={counter} samples)\n"
                f"Lower = more sensitive to weights = better",
                y=1.02, fontsize=13, fontweight='bold'
            )

            out_path = f"{base}_{metric}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")
            
    # ------------------------------------------------------------------
    # Similarity metrics
    # ------------------------------------------------------------------

    
    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def visualize_comparison(
        original_img, heatmaps_dict,
        title_prefix="", alpha=0.6, save_path=None
    ):
        """
        Shows original + all heatmaps.
        If more than 6 methods, uses two rows.
        """
        if isinstance(original_img, torch.Tensor):
            if original_img.ndim == 4:
                original_img = original_img.squeeze(0)
            img_np = original_img.permute(1, 2, 0).detach().cpu().numpy()
        else:
            img_np = original_img

        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        h, w   = img_np.shape[:2]

        valid      = {k: v for k, v in heatmaps_dict.items() if v is not None}
        n_methods  = len(valid)
        max_per_row = 6

        if n_methods > max_per_row:
            # Two rows — first row includes input image
            items      = list(valid.items())
            row1_items = items[:max_per_row]
            row2_items = items[max_per_row:]
            n_cols     = max(max_per_row + 1, len(row2_items))  # +1 for input image

            fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 10))

            # Row 0: input + first batch of methods
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')

            for i, (name, hmap) in enumerate(row1_items, start=1):
                ax = axes[0, i]
                if hmap.shape[:2] != (h, w):
                    hmap = cv2.resize(hmap, (w, h))
                hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
                ax.imshow(img_np)
                im = ax.imshow(hmap_norm, cmap='jet', alpha=alpha)
                ax.set_title(f"{title_prefix}{name}", fontsize=8)
                ax.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

            # Hide unused axes in row 0
            for j in range(len(row1_items) + 1, n_cols):
                axes[0, j].axis('off')

            # Row 1: second batch of methods
            for i, (name, hmap) in enumerate(row2_items):
                ax = axes[1, i]
                if hmap.shape[:2] != (h, w):
                    hmap = cv2.resize(hmap, (w, h))
                hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
                ax.imshow(img_np)
                im = ax.imshow(hmap_norm, cmap='jet', alpha=alpha)
                ax.set_title(f"{title_prefix}{name}", fontsize=8)
                ax.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

            # Hide unused axes in row 1
            for j in range(len(row2_items), n_cols):
                axes[1, j].axis('off')

        else:
            # Single row — original layout
            num_plots = n_methods + 1
            fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 4, 5))
            if num_plots == 1:
                axes = [axes]

            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[0].axis('off')

            for i, (name, hmap) in enumerate(valid.items()):
                ax = axes[i + 1]
                if hmap.shape[:2] != (h, w):
                    hmap = cv2.resize(hmap, (w, h))
                hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
                ax.imshow(img_np)
                im = ax.imshow(hmap_norm, cmap='jet', alpha=alpha)
                ax.set_title(f"{title_prefix}{name}", fontsize=9)
                ax.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

        plt.tight_layout()
        os.makedirs("PAF-output", exist_ok=True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return fig

    def _autolabel(self, ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy         = (rect.get_x() + rect.get_width() / 2, height),
                xytext     = (0, 3),
                textcoords = "offset points",
                ha='center', va='bottom', fontsize=7, rotation=45
            )


def run_randomization_tests(
        test_loader, 
        model, 
        model_name,
        device,
        n_samples,
        graph_manager,
        paf_modes,
        output_dir="PAF-output",
        random_sample:bool = True
):
    all_results = {}  
    method_names = ["PAF", "GCAM", "IG", "LRP","DEEPSHAP"]
    collected = 0
    sample_id=0
    rand_test=RandomizationTestMultiMode(
        model=model,
        device=device,
        paf_modes=paf_modes,
        model_name=model_name,
        graph_manager=graph_manager
        )
    all_samples=[]
    while collected < n_samples:
        test_sample = get_sample_image(
            test_loader, model, device,
            random_sample,
            samples_checked=sample_id
        )
        idx, x, y, predicted, sample_id = test_sample
        true_label = y.item() if isinstance(y, torch.Tensor) else y
        all_samples.append({"idx":idx,"x":x,"y":true_label,"predicted":predicted, "sample_id":sample_id})
        collected += 1

    labels = ResNet18_Weights.DEFAULT.meta["categories"]
    results=rand_test.run(all_samples,labels,output_dir+"/"+model_name+".png",test_type='both')
    rand_test.cleanup()
    del rand_test
    del test_sample
    return results

def load_and_plot_all(output_dir):
    # 1. Gather all CSV files in the checkpoint directory
    path = output_dir
    all_files = glob.glob(os.path.join(path, "*.csv"))

    if not all_files:
        print("No results found!")
        return

    # 2. Combine them into one big DataFrame
    df_list = [pd.read_csv(f) for f in all_files]
    final_df = pd.concat(df_list, ignore_index=True)
    
    print(f"Loaded results for {final_df['model_name'].nunique()} models.")
    
    # 3. Save the final master file for the paper
    final_df.to_csv("final_randomization_results_master.csv", index=False)
    
    return final_df

def plot_specific_randomization(df, test_type="weight"):
    # Filter for the specific test
    sub_df = df[df['test_type'] == test_type]
    
    # Melt to handle Spearman and SSIM as sub-plots
    df_melted = sub_df.melt(
        id_vars=['method', 'stage'], 
        value_vars=['spearman', 'ssim'], 
        var_name='metric', value_name='similarity'
    )

    sns.set_theme(style="whitegrid", font="serif")
    
    # Create a 1x2 grid (Spearman on left, SSIM on right)
    g = sns.FacetGrid(
        df_melted, col='metric', hue='stage', 
        height=5, aspect=1.2, sharey=True,
        palette={'last_conv': '#66c2a5', 'full': '#fc8d62'}
    )

    g.map_dataframe(
        sns.boxplot, x='method', y='similarity', 
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white"}
    )

    # Styling
    title_suffix = "Weight (Parametric)" if test_type == "weight" else "Activation (Data)"
    g.figure.suptitle(f"Sanity Check: {title_suffix}", fontweight='bold', y=1.05)
    
    for ax in g.axes.flat:
        ax.axhline(0.3, ls='--', color='red', alpha=0.6, label='Threshold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    g.add_legend()
    plt.savefig(f"PAF-output/{test_type}_randomization_final_results.pdf", bbox_inches='tight')
    plt.show()

