"""
PAF Evaluation Runner - Command-line interface for PAF experiments.

Supports four types of evaluations:
1. perturbation - Deletion/Insertion (AUDC/AUIC) tests
2. randomization - Weight randomization sanity checks
3. qualitative - Visualization and layer-wise analysis
4. pointing_game - Bounding box localization evaluation

Usage:
    python paf_evaluation.py --test-type perturbation --model resnet18 --dataset imagenet
    python paf_evaluation.py --test-type randomization --num-samples 100
    python paf_evaluation.py --test-type qualitative --visualize-layers
    python paf_evaluation.py --test-type pointing_game --max-samples 500
"""

from collections import defaultdict
import os
import sys
from pathlib import Path

import pandas as pd

'''
# Add src/ to Python path to enable module imports
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up 3 levels to project root
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
'''
import argparse
import torch
import random
import numpy as np
import yaml

# Ensure Evaluation.utils_main is imported first to fix sys.path shadowing issues
from core.model.io import load_model
from core.paf.paf import PAF
from core.paf.scoring import ScoringMode
from core.model.config import ModelConfig, DataConfig
from core.model.factory import ModelFactory
from core.paf.graph.manager import PAFGraphManager
from core.paf.utils import parse_mode_from_string
from core.visualization.visualizer import PAFVisualizer
from Evaluation.randomization_test_multimode import run_randomization_tests, load_and_plot_all, plot_specific_randomization 
from Evaluation.perturbation_test_multimode import plot_neurips_aggregate, generate_summary_table, run_perturbation_tests
from Evaluation.pointing_game import _aggregate_results, _print_results, evaluate_pointing_game
from Evaluation.utils_main import (
    load_config, load_model_dataset, get_test_sample, resolve_device, _paf_mode_key
)
from Evaluation.eval_core.metrics import compute_aggregate_statistics


def parse_paf_modes(mode_config):
    """
    Parse PAF modes from config or command line.

    Parameters
    ----------
    mode_config : list of str or dict
        Mode specifications: ["ABS:tau=1.0", "POWER:tau=2.0"] or config dict

    Returns
    -------
    list of tuple
        [(ScoringMode.ABS, {'tau': 1.0}), ...]
    """
    modes = []

    if isinstance(mode_config, list):
        for mode_spec in mode_config:
            if isinstance(mode_spec, str):
                parts = mode_spec.split(':')
                mode_name = parts[0].upper()
                mode = ScoringMode[mode_name]

                # Parse parameters
                params = {}
                if len(parts) > 1:
                    for param in parts[1:]:
                        key, val = param.split('=')
                        params[key] = float(val) if '.' in val else int(val)

                modes.append((mode, params))

    return modes if modes else [(ScoringMode.ABS, {'tau': 1.0})]


class PAFEvaluator:
    """Main PAF evaluation runner."""

    def __init__(self, args):
        """Initialize evaluator with parsed arguments."""
        self.args = args
        self.device = resolve_device(args.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds for reproducibility
        if args.seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        # Load model and dataset
        #print(f"Loading model: {args.model}")
        #self.model, self.graph_manager, self.test_loader, self.config_data, _ = load_model_dataset()
        #self.model = self.model.to(self.device)

        # Parse PAF modes
        #self.paf_modes = parse_paf_modes(
        #    args.paf_modes if args.paf_modes else
        #    [(ScoringMode.ABS, {'tau': 1.0})]
        #)

        #print(f"PAF Modes: {self.paf_modes}")
        #print(f"Device: {self.device}")

    def perturbation_test(self):
        """
        Run Deletion/Insertion (perturbation) tests.
        Measures: AUDC (Area Under Deletion Curve), AUIC (Area Under Insertion Curve)
        """
        print("\n" + "="*80)
        print("PERTURBATION TEST - Deletion/Insertion Evaluation")
        print("="*80)

        device = self.device or resolve_device("auto")
        config = load_config("config/evaluation_config.yaml")
       
        model_names = self.args.model if self.args.model is not None else config.get('model', {}).get('name', ['resnet18'])
        models_config_path= config.get('model', {}).get('model_config_path', 'config/models_config.yaml')
        
        dataset_name = self.args.dataset or config.get('dataset', {}).get('name') or 'imagenet'
        data_path=self.args.data_path or config.get('dataset', {}).get('path') or './data/imagenet-1k'
        
        num_classes= config.get('dataset', {}).get('num_classes', 1000)
        steps=self.args.steps or config.get('benchmarking', {}).get('steps', 10)
        num_samples=self.args.num_samples or config.get('benchmarking', {}).get('num_samples', 10)
        shuffle = config.get('experiment', {}).get('random_sample', False)
        batch_size = config.get('dataset', {}).get('batch_size', 1)
        parse = lambda arg: (
            parse_mode_from_string(arg.partition(':')[0]),
            dict(
                (p.split('=', 1)[0], float(p.split('=', 1)[1])) 
                for p in arg.partition(':')[2].split(',') 
                if '=' in p
            )
        )
        paf_mode_args = [parse(m) for m in self.args.paf_modes] if self.args.paf_modes else None        
        paf_modes = paf_mode_args or [
                (parse_mode_from_string(entry.get('mode')), entry.get('params', {})) 
                for entry in config.get('paf_modes', [])
            ]
        global_metrics = {}

        for model_name in model_names:
            print(f"\n{'='*20}\nEvaluating Model: {model_name}\n{'='*20}")

            model_cfg = ModelConfig(
                        model_name=model_name,
                        num_classes=num_classes,
                        pretrained=True,
                        device=str(device),
                        models_config_path=models_config_path,
                        dataset=dataset_name
                    )
            data_cfg = DataConfig(
                        data_path=data_path,
                        dataset=dataset_name,
                        batch_size=batch_size,
                        shuffle=shuffle
                    )

            factory = ModelFactory(models_config_path)
            model = factory.create_model(model_cfg).to(device)
            model.eval()
            graph_manager = PAFGraphManager(model)
            
            print("Loading Validation samples...")
            test_loader = factory.get_dataloader(data_cfg, model_cfg)
            steps, results = run_perturbation_tests(
                test_loader=test_loader, 
                model=model, 
                device=device, 
                n_samples=num_samples, 
                model_name=model_name, 
                graph_manager=graph_manager,
                paf_modes=paf_modes
                )

            for method, res in results.items():
                if method not in global_metrics:
                    global_metrics[method] = {'ins': [], 'del': []}
                global_metrics[method]['ins'].extend(res['ins'])
                global_metrics[method]['del'].extend(res['del'])

        method_names = list(global_metrics.keys())
        stats, audc, auic, del_matrix, ins_matrix = compute_aggregate_statistics(
                                                        global_metrics, method_names, steps
                                                    )
        plot_neurips_aggregate(steps, del_matrix, ins_matrix, audc, auic)
        generate_summary_table(stats, audc, auic, method_names, output_dir=self.output_dir,filename="summary_table.txt")
        print(f"✓ Perturbation test completed. Results")
       

    def randomization_test(self):
        """
        Run randomization (sanity check) tests.
        Randomizes model weights at different stages and checks attribution consistency.
        """
        print("\n" + "="*80)
        print("RANDOMIZATION TEST - Weight and Activation Randomization Sanity Check")
        print("="*80)

        config = load_config("config/evaluation_config.yaml")

        model_names = self.args.model if self.args.model is not None else config.get('model', {}).get('name', ['resnet18'])
        models_config_path = config.get('model', {}).get('model_config_path', 'config/models_config.yaml')
        dataset_name = self.args.dataset or config.get('dataset', {}).get('name', 'imagenet')
        data_path = self.args.data_path or config.get('dataset', {}).get('path', './data/imagenet-1k')
        num_classes = config.get('dataset', {}).get('num_classes', 1000)
        num_samples = self.args.num_samples or config.get('benchmarking', {}).get('num_samples', 1)
        random_sample = config.get('experiment', {}).get('random_sample', False)
        sample_idx = self.args.sample_idx if self.args.sample_idx is not None else config.get('dataset', {}).get('sample_idx', 0)

        paf_mode_args = [parse_paf_modes(self.args.paf_modes)] if self.args.paf_modes else None
        paf_modes = paf_mode_args[0] if paf_mode_args else [
            (parse_mode_from_string(entry.get('mode')), entry.get('params', {}))
            for entry in config.get('paf_modes', [])
        ]
        results = {}

        for model_name in model_names:
            print(f"\n{'='*20}\nRandomization evaluation for model: {model_name}\n{'='*20}")

            model_cfg = ModelConfig(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=True,
                device=str(self.device),
                models_config_path=models_config_path,
                dataset=dataset_name
            )
            data_cfg = DataConfig(
                data_path=data_path,
                dataset=dataset_name,
                batch_size=1,
                shuffle=random_sample
            )

            factory = ModelFactory(models_config_path)
            model = factory.create_model(model_cfg).to(self.device)
            model.eval()
            graph_manager = PAFGraphManager(model)
            print("Loading validation samples...")
            test_loader = factory.get_dataloader(data_cfg, model_cfg)

            visualize_heatmap = self.args.visualize_heatmap
            res = run_randomization_tests(
                test_loader=test_loader,
                model=model,
                model_name=model_name,
                device=self.device,
                n_samples=num_samples,
                graph_manager=graph_manager,
                paf_modes=paf_modes,
                output_dir=str(self.output_dir),
            )
            results[model_name] = res

            rows = []
            for tt, entries in results[model_name].items():
                for entry in entries:
                    row = entry.copy()
                    row['test_type'] = tt
                    row['model_name'] = model_name
                    rows.append(row)
            
            # Save this model's specific data
            df_temp = pd.DataFrame(rows)
            os.makedirs(self.output_dir, exist_ok=True)
            df_temp.to_csv(f"{self.output_dir}/{model_name}_results.csv", index=False)
            print(f"Checkpointed results for {model_name}")

        df=load_and_plot_all(self.output_dir)
        # Clean up: Ensure numeric types for metrics
        df['spearman'] = pd.to_numeric(df['spearman'], errors='coerce')
        df['ssim'] = pd.to_numeric(df['ssim'], errors='coerce')
        weight_df = df[df['test_type'] == 'weight']
        activation_df = df[df['test_type'] == 'activation']
        plot_specific_randomization(weight_df, test_type="weight")
        plot_specific_randomization(activation_df, test_type="activation")

        print(f"✓ Randomization test completed for models: {list(results.keys())}")

    def qualitative_test(self):
        """
        Run qualitative visualization tests.
        Generates heatmaps, layer visualizations, and comparative analysis.
        """
        print("\n" + "="*80)
        print("QUALITATIVE TEST - Visualization & Layer Analysis")
        print("="*80)

        print("\n" + "="*80)
        print("RANDOMIZATION TEST - Weight and Activation Randomization Sanity Check")
        print("="*80)

        config = load_config("config/evaluation_config.yaml")

        model_names = self.args.model if self.args.model is not None else config.get('model', {}).get('name', ['resnet18'])
        models_config_path = config.get('model', {}).get('model_config_path', 'config/models_config.yaml')
        dataset_name = self.args.dataset or config.get('dataset', {}).get('name', 'imagenet')
        data_path = self.args.data_path or config.get('dataset', {}).get('path', './data/imagenet-1k')
        num_classes = config.get('dataset', {}).get('num_classes', 1000)
        
        #num_samples = self.args.num_samples or config.get('benchmarking', {}).get('num_samples', 1)
        random_sample = config.get('experiment', {}).get('random_sample', False)
        sample_idx = self.args.sample_idx if self.args.sample_idx is not None else config.get('dataset', {}).get('sample_idx', 0)

        paf_mode_args = [parse_paf_modes(self.args.paf_modes)] if self.args.paf_modes else None
        paf_modes = paf_mode_args[0] if paf_mode_args else [
            (parse_mode_from_string(entry.get('mode')), entry.get('params', {}))
            for entry in config.get('paf_modes', [])
        ]
        if len(model_names) < 1:
            raise ValueError("No model specified for qualitative test. Use --model or check config.")


        model_name=model_names[0]
        mode=paf_modes[0]

        model_cfg = ModelConfig(
                model_name=model_names[0],  # only the first model for qualitative test
                num_classes=num_classes,
                pretrained=True,
                device=str(self.device),
                models_config_path=models_config_path,
                dataset=dataset_name
            )
        
        data_cfg = DataConfig(
            data_path=data_path,
            dataset=dataset_name,
            batch_size=1,
            shuffle=random_sample
        )

        factory = ModelFactory(models_config_path)
        model = factory.create_model(model_cfg).to(self.device)
        model.eval()
        graph_manager = PAFGraphManager(model)
        print("Loading validation samples...")
        test_loader = factory.get_dataloader(data_cfg, model_cfg)


        test_sample = get_test_sample(test_loader, model, self.device)
        idx, x, y, predicted, sample_id = test_sample

        # Create PAF instance
        paf = PAF(
            model=model,
            graph_manager=graph_manager,
            debug_level=self.args.debug_level,
            x=x,
            target_class=y.item() if isinstance(y, torch.Tensor) else y,
            true_class=y.item() if isinstance(y, torch.Tensor) else y,
            modes=paf_modes,
        )

        visualizer = PAFVisualizer(
            paf,
            misclassification=self.args.analyze_misclassification,
            contrastive_explanation=self.args.contrastive,
            debug_level=self.args.debug_level,
            true_class=y.item() if isinstance(y, torch.Tensor) else y,
            target_class=predicted,
        )

        # Generate visualizations
        output_path = self.output_dir / f"qualitative_sample_{sample_id}.png"

        if self.args.visualize_layers:
            print(f"Generating layer-wise visualizations...")
            visualizer.visualize_layerwise_heatmaps(
                mode_key=_paf_mode_key(self.paf_modes[0]),
                cols=8,
                save_path=str(output_path),
            )

        if self.args.visualize_canny:
            print(f"Generating Canny edge alignment visualizations...")
            output_path_canny = self.output_dir / f"qualitative_canny_{sample_id}.png"
            visualizer.visualize_canny_evolution(
                mode_key=_paf_mode_key(self.paf_modes[0]),
                cols=8,
                save_path=str(output_path_canny),
            )

        # Main figure
        output_path_main = self.output_dir / f"qualitative_main_{sample_id}.pdf"
        visualizer.visualize_nips_killer_figure(
            x,
            y.item() if isinstance(y, torch.Tensor) else y,
            save_path=str(output_path_main),
        )

        #visualizer.show_salient_features(x, mode, save_path=str(self.output_dir)+"/salient_" + model_name + "_" + dataset_name + "_" + str(sample_id) + ".png")
        #visualizer.plot_salient_contours(x, mode, save_path=str(self.output_dir)+"/contours_" + model_name + "_" + dataset_name + "_"  + str(sample_id) + ".png")
        #visualizer.show_clean_contours(x, mode, save_path=str(self.output_dir)+"/clean_contours_" + model_name + "_" + dataset_name + "_" + str(sample_id) + ".png")
        print(f"✓ ImageNet Sample {sample_id} analyzed. Prediction: {predicted}, Actual: {y}")
        visualizer.visualize_layer_results(x, mode, save_path=str(self.output_dir)+"/layer_results_" + model_name + "_" + dataset_name + "_"  + str(sample_id))
        visualizer.plot_paf_branching_logic(is_signed=False,save_path=str(self.output_dir)+"/paf_branching_" + model_name + "_" + dataset_name + "_" + str(sample_id) + ".png")

        print(f"✓ Qualitative visualizations saved to {self.output_dir}")
        return True

    def pointing_game_test(self):
        """
        Run pointing game (bounding box localization) evaluation.
        Measures how well heatmaps localize objects in bounding boxes.
        """
        print("\n" + "="*80)
        print("POINTING GAME - Bounding Box Localization Evaluation")
        print("="*80)

        # This requires a pointing game dataloader (VOC or ImageNet with boxes)
        try:
            config = load_config("config/evaluation_config.yaml")
            model_names = self.args.model if self.args.model is not None else config.get('model', {}).get('name', ['resnet18'])
            models_config_path = config.get('model', {}).get('model_config_path', 'config/models_config.yaml')
            num_classes = config.get('dataset', {}).get('num_classes', 1000)
            num_samples = self.args.max_samples or config.get('benchmarking', {}).get('num_samples', 1)
            random_sample = config.get('experiment', {}).get('random_sample', False)
            dataset_name = self.args.dataset or config.get('dataset', {}).get('name', 'imagenet')
            data_path = self.args.data_path or config.get('dataset', {}).get('path', './data/imagenet-1k')
            parse = lambda arg: (
                parse_mode_from_string(arg.partition(':')[0]),
                dict(
                    (p.split('=', 1)[0], float(p.split('=', 1)[1])) 
                    for p in arg.partition(':')[2].split(',') 
                    if '=' in p
                )
            )
            paf_mode_args = [parse(m) for m in self.args.paf_modes] if self.args.paf_modes else None        
            paf_modes = paf_mode_args or [
                    (parse_mode_from_string(entry.get('mode')), entry.get('params', {})) 
                    for entry in config.get('paf_modes', [])
                ]
            
            thresholds  = [0.5, 0.6, 0.7]

            # Shared accumulators across all models
            all_hits        = defaultdict(int)
            all_total       = defaultdict(int)
            all_mass_scores = defaultdict(list)
            all_th_hits     = {t: defaultdict(int) for t in thresholds}

            for model_name in model_names:
                print(f"\n{'='*20}\nRandomization evaluation for model: {model_name}\n{'='*20}")

                model_cfg = ModelConfig(
                    model_name=model_name,
                    num_classes=num_classes,
                    pretrained=True,
                    device=str(self.device),
                    models_config_path=models_config_path,
                    dataset=dataset_name
                )
                data_cfg = DataConfig(
                    data_path=data_path,
                    dataset=dataset_name,
                    batch_size=1,
                    shuffle=random_sample
                )

                factory = ModelFactory(models_config_path)
                model = factory.create_model(model_cfg).to(self.device)
                model.eval()
                graph_manager = PAFGraphManager(model)
                print("Loading validation samples...")
                if dataset_name == "voc":
                    from Evaluation.VOC_bounding_box_dataset import get_voc_pointing_game_loader
                    loader = get_voc_pointing_game_loader(
                        root=self.args.data_path,
                        max_samples=self.args.max_samples,
                    )
                else:
                    print("Note: Pointing game typically uses VOC dataset. Using validation loader instead.")
                    # Load model and dataset
                    dataset_name = self.args.dataset or config.get('dataset', {}).get('name', 'imagenet')
                    data_path = self.args.data_path or config.get('dataset', {}).get('path', './data/imagenet-1k')
                    loader = factory.get_dataloader(data_cfg, model_cfg)

                hits, total, mass_scores, th_hits  = evaluate_pointing_game(
                    paf_class=PAF,
                    model=model,
                    graph_manager=graph_manager,
                    dataloader=loader,
                    device=self.device,
                    paf_modes=paf_modes,
                    num_samples=num_samples,
                    thresholds=thresholds,
                )
                print(f"✓ Pointing game evaluation completed for model: {model_name}.")
                for method, count in hits.items():
                    all_hits[method]  += count
                for method, count in total.items():
                    all_total[method] += count
                for method, scores in mass_scores.items():
                    all_mass_scores[method].extend(scores)
                for t in thresholds:
                    for method, count in th_hits[t].items():
                        all_th_hits[t][method] += count
                    
            # Print aggregated results across ALL models — reuse existing helpers
            # Aggregate once at the end
            print(f"\n{'='*40}")
            print(f"AGGREGATED RESULTS — {len(model_names)} models")
            print(f"{'='*40}")
            final_results = _aggregate_results(
                all_hits, all_total, all_mass_scores, all_th_hits, thresholds
            )
            _print_results(final_results, thresholds)
            return final_results

        except Exception as e:
            print(f"⚠ Pointing game evaluation failed: {e}")
            print(f"  This test requires VOC dataset. Ensure --data-path points to VOC root.")
            return None


def main():
    """Main entry point for PAF evaluation."""
    parser = argparse.ArgumentParser(
        description="PAF Evaluation Suite - Run attribution evaluation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run perturbation test with 50 steps
  python paf_evaluation.py --test-type perturbation --steps 50

  # Run randomization test with 100 iterations
  python paf_evaluation.py --test-type randomization --num-samples 100

  # Run qualitative visualization with layer analysis
  python paf_evaluation.py --test-type qualitative --visualize-layers --visualize-canny

  # Run pointing game with VOC dataset
  python paf_evaluation.py --test-type pointing_game --dataset-type voc --max-samples 500

  # Test multiple PAF modes
  python paf_evaluation.py --test-type perturbation --paf-modes ABS:tau=1.0 POWER:tau=2.0 NORM:tau=1.0
        """
    )

    # Test type selection
    parser.add_argument(
        '--test-type', type=str, required=True,
        choices=['perturbation', 'randomization', 'qualitative', 'pointing_game'],
        help='Type of evaluation to run'
    )

    # Model and dataset
    parser.add_argument(
        '--model', type=str, nargs='*', default=None,
        help='Model(s) name (default: None)'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset name ((e.g., imagenet, voc, cifar10), default: None)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to dataset root (default: None)'
    )
    
    # Test-specific parameters
    parser.add_argument(
        '--steps', type=int, default=10,
        help='Steps for perturbation test (default: 10)'
    )
    parser.add_argument(
        '--patch-size', type=int, default=1,
        help='Patch size for perturbation (default: 1)'
    )
    parser.add_argument(
        '--num-samples', type=int, default=5,
        help='Number of samples/iterations for test (default: 5)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=50,
        help='Max samples for pointing game (default: 100)'
    )
    parser.add_argument(
        '--sample-idx', type=int, default=0,
        help='Specific sample index to analyze (default: 0 = random)'
    )

    # PAF modes
    parser.add_argument(
        '--paf-modes', type=str, nargs='+',
        default=None,
        help='PAF modes to test (format: MODE:param=value, Available: ABS:tau=1.0, POWER:tau=n.0, NORM:tau=1.0 NORM-POWER:tau=n.0, n>0). Default: None (uses config or defaults to ABS:tau=1.0)'
    )

    # Visualization options
    parser.add_argument(
        '--visualize-layers', action='store_true',
        help='Generate layer-wise heatmaps (qualitative only)'
    )
    parser.add_argument(
        '--visualize-canny', action='store_true',
        help='Generate Canny edge alignment visualizations (qualitative only)'
    )
    parser.add_argument(
        '--visualize-heatmap', action='store_true',
        help='Visualize heatmaps during test'
    )

    # Analysis options
    parser.add_argument(
        '--analyze-misclassification', action='store_true',
        help='Analyze misclassifications instead of correct predictions'
    )
    parser.add_argument(
        '--contrastive', action='store_true',
        help='Use contrastive interpretation'
    )
    parser.add_argument(
        '--use-baselines', action='store_true',
        help='Include baseline attribution methods (Captum, GradCAM) for comparison'
    )

    # System options
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./PAF-output',
        help='Output directory for results (default: ./PAF-output)'
    )
    parser.add_argument(
        '--debug-level', type=int, default=0,
        help='Debug verbosity level (0=silent, 1=info, 2=verbose)'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = PAFEvaluator(args)

    if args.test_type == 'perturbation':
        result = evaluator.perturbation_test()
    elif args.test_type == 'randomization':
        result = evaluator.randomization_test()
    elif args.test_type == 'qualitative':
        result = evaluator.qualitative_test()
    elif args.test_type == 'pointing_game':
        result = evaluator.pointing_game_test()

    print("\n" + "="*80)
    print("✓ PAF Evaluation Complete")
    print("="*80)


if __name__ == '__main__':
    main()
