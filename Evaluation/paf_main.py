import argparse
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights
from pathlib import Path
import sys
from scipy import stats
import pandas as pd

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from Evaluation.VOC_bounding_box_dataset import get_voc_pointing_game_loader
from Evaluation.utils_main import _paf_mode_key
from PAF.paf import *
from model_factory import ModelConfigLoader, ModelFactory, TrainingConfig
from nn_arch.paf_hook_manager import PAFHookManager
from PAF.paf_visualizer import PAFVisualizer
from Evaluation.perturbation_test import *
from Evaluation.randomization_test import *
from Evaluation.randomization_test_multimode import *
from Evaluation.perturbation_test_multimode import *
from Evaluation.utils_main import *
import warnings
from Evaluation.pointing_game import evaluate_pointing_game
from Evaluation.bounding_box_dataset import *
warnings.filterwarnings("ignore",category=UserWarning,module="captum")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels=ResNet18_Weights.DEFAULT.meta["categories"]



def run_quantitative_suite(args):
    """Runs Deletion and Insertion tests and prints AUDC/AUIC results."""
    print(f"--- Running Quantitative Suite (Steps: {args.steps}, Patch: {args.patch_size}) ---")
    # your_data_loader logic here
    # results = run_patch_deletion_experiment(...)
    pass

def run_localization_suite(args):
    """Runs Energy-based Pointing Game and Canny Alignment."""
    print("--- Running Localization & Edge Alignment Suite ---")
    # ebpg_score = calculate_ebpg(...)
    pass

def run_sanity_checks(args):
    """Performs weight randomization tests."""
    print("--- Running Model Parameter Randomization (Adebayo Sanity Check) ---")
    pass

def run_paf(model, model_name,test_samples,paf_modes,debug_level,dataset_name):
    
    idx, x, y, predicted, sample_id = test_samples
    true_label = y.item() if isinstance(y, torch.Tensor) else y
    hook_manager=PAFHookManager(model)
    '''
    for pmode in paf_modes:
        (mode,params)=pmode
        mode_name="PAF-"+mode.value
        key=_paf_mode_key(pmode)
    '''
    print(f"Label: {labels[y]}")
    paf=PAF(
        model=model,
        hook_manager=hook_manager,
        debug_level=debug_level,
        x=x,
        target_class=true_label,true_class=true_label,
        modes       = paf_modes
        )
    visualizer = PAFVisualizer(paf, true_class=true_label, target_class=predicted)
    ##visualizer.visualize_layer_results(x, key,save_path="PAF-output/" + model_name + "_" + dataset_name + "_" + mode_name + str(sample_id))
    #visualizer.visualize_all_layers_grid(x, key,save_path="PAF-output/" + model_name + "_" + dataset_name + "_" + mode_name + str(sample_id))
    #visualizer.visualize_layerwise_heatmaps(key,8,"PAF-output/heatmaps.png")
    #visualizer.visualize_canny_evolution(mode_key=key,cols=8,save_path="PAF-output/heatmaps_canny.png")
    #visualizer.visualize_nips_killer_figure(x,true_label,save_path="PAF-output/heatmaps_canny.pdf")
    #loader=get_imagenet_pointing_game_loader("./data/imagenet-1k")
    
    '''
    loader = get_imagenet_csv_loader(
        imagenet_root = './data/imagenet-1k',
        csv_path      = './data/imagenet-1k/LOC_val_solution.csv',
        batch_size    = 1,
        max_samples   = 1000,
        num_workers   = 0,   # 0 for MPS to avoid multiprocessing issues
    )
    '''

    loader = get_voc_pointing_game_loader(
    root        = './data/voc',   # downloads automatically
    max_samples = 1000,
)

    results = evaluate_pointing_game(
        paf_class          = PAF,
        model              = model,
        hook_manager       = hook_manager,
        dataloader         = loader,   # yields (images, labels, boxes)
        device             = device,
        paf_modes          = paf_modes,
        target_layer_name  = 'features_28', #'layer4_1_conv2',
        num_samples        = 300,
        use_baselines      = True,
    )
    print(results)

def main():
    '''
    parser = argparse.ArgumentParser(description="PAF Experiment Runner for NeurIPS Submission")

    # Mode Selection
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['quant', 'loc', 'sanity', 'all','viz-layer'],
                        help="Choose experiment: quant (Del/Ins), loc (EBPG), sanity (Checks), or all")

    # Parameters
    parser.add_argument('--steps', type=int, default=10, help="Number of steps for Del/Ins")
    parser.add_argument('--patch_size', type=int, default=3, help="Patch size (e.g., 3 for 3x3)")
    parser.add_argument('--eps', type=float, default=1e-2, help="Epsilon stabilizer for PAF")
    parser.add_argument('--output_dir', type=str, default='./PAF-output', help="Where to save plots/csv")

    args = parser.parse_args()
    '''

    model, hook_manager, test_loader, config_data=load_model_dataset()
    
    #test_sample=get_test_sample(test_loader, model, device)
    test_sample=modify_testsample(test_loader.dataset[27026],model)
    model_name= config_data['model']['name']
    debug_level=config_data['experiment']['debug_level']
    dataset_name=config_data['dataset']['name']
    paf_modes = []
    for entry in config_data['paf_modes']: 
        mode=entry['mode'] 
        mode = getattr(ScoringMode, mode.split('.')[-1])
        params=entry['params']
        paf_modes.append((mode,params))

    run_paf(model,model_name,test_sample,paf_modes,debug_level,dataset_name)

    '''
    # Route to the correct experiment
    if args.mode == 'quant' or args.mode == 'all':
        run_quantitative_suite(args)
    
    if args.mode == 'loc' or args.mode == 'all':
        run_localization_suite(args)
        
    if args.mode == 'sanity' or args.mode == 'all':
        run_sanity_checks(args)
    '''
if __name__ == "__main__":
    main()