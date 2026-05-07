import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights
from pathlib import Path
import sys
from scipy import stats
import pandas as pd


parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from PAF.paf import *
from model_factory import ModelConfigLoader, ModelFactory, TrainingConfig
from nn_arch.paf_hook_manager import PAFHookManager
from PAF.paf_visualizer import PAFVisualizer
from Evaluation.perturbation_test import *
from Evaluation.randomization_test import *
from Evaluation.randomization_test_multimode import *
from Evaluation.perturbation_test_multimode import *
import warnings

comparative_analysis=True
#model_name="vit_b_16"
model_name="resnet18"
#dataset_name="imagenette"           # Change to "imagenet" if using the full ImageNet dataset
dataset_name="imagenet"
#dataset_path='./data/imagenette'
dataset_path="./data/imagenet-1k"
yaml_config_path="models/models_config.yaml"
analyze_misclassification=False   # Set to False to analyze mispredictions instead
contrastive_interpretation=True
random_sample=True
sample_idx=0    
test_runs=100               # Batch Index of the sample to analyze 
patch_size=3                 # insertion/deletion, area of insertion deletion
test_name="randomization"
debug_level=0
analyze_multimode=True      #Temporary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ResNet18_Weights.DEFAULT.meta["categories"]
paf_modes=[
        (ScoringMode.ABS,   {'tau': 1.0}),
        (ScoringMode.POWER, {'tau': 2.0}),
        #(ScoringMode.POWER, {'tau': 0.5}),
        #(ScoringMode.POWER, {'tau': 5.0}),
        #(ScoringMode.EXP_WEIGHT,{'tau': 3.0}),
        (ScoringMode.NORM,{'tau': 1.0}),
        (ScoringMode.NORM_POWER,{'tau': 2.0}),  
        #(ScoringMode.SIGNED_SPLIT,{'tau': 1.0, 'alpha': 0.7, 'beta': 0.3}),  # exhibitory
        #(ScoringMode.SIGNED_SPLIT,{'tau': 1.0, 'alpha': 0.0, 'beta': 1.0}),  # inhibitory
        #(ScoringMode.SIGNED_FULL, {'tau': 1.0}),           # directional analysis       
        ]

warnings.filterwarnings("ignore",category=UserWarning,module="captum")

def tensor_to_display(x_tensor):
    # 1. Remove batch dim and move to CPU
    # x shape: [1, 3, 224, 224] -> [3, 224, 224]
    img = x_tensor.squeeze().cpu().detach()
    
    # 2. Inverse ImageNet Normalization
    # These are the standard ImageNet values used during training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = img * std + mean # Multiply by std, then add mean
    
    # 3. Clip to [0, 1] range to avoid floating point errors
    img = torch.clamp(img, 0, 1)
    
    # 4. Permute to [H, W, C] for Matplotlib
    return img.permute(1, 2, 0).numpy()

def get_refined_lrp(model, input_tensor, target_class=0):
    # 1. Initialize and Attribute
    from captum.attr import LRP
    # 1. Initialize LRP
    # Note: If this still looks dotty, it is because ResNet-18 
    # needs specific LRP-Epsilon or LRP-Gamma composites.
    lrp = LRP(model)
    
    # 2. Attribute
    attribution = lrp.attribute(input_tensor, target=target_class)
    
    # 3. Collapse Channels (Sum) and remove negative relevance
    # We use sum because we want the total evidence supporting the class
    raw_map = attribution.squeeze().sum(dim=0).cpu().detach().numpy()
    raw_map = np.maximum(raw_map, 0)
    
    # 4. POWER SCALING (Crucial for ResNet LRP)
    # Raising to a power < 1 (like 0.5) amplifies weak signals 
    # and helps 'connect' the sparse dots into a shape.
    processed_map = np.power(raw_map, 0.5)
    
    # 5. PERCENTILE CLIPPING (Removes outliers that make the rest look dim)
    v_max = np.percentile(processed_map, 99) # Focus on top 1% signals
    processed_map = np.clip(processed_map, 0, v_max) / (v_max + 1e-8)
    
    # 6. GAUSSIAN BLUR
    # This simulates the 'flow' aspect of your PAF tool
    # Sigma 2.0-3.0 is best for cleaning up ResNet noise
    final_heatmap = gaussian_filter(processed_map, sigma=2.5)
    
    return final_heatmap

def gradcam_visualization(model, x, pred_val, sample_idx=0):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import torch.nn as nn
    import os
    # Fix inplace ReLU before anything else
    def replace_inplace_relu(m):
        for name, module in m.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                parts = name.split('.')
                parent = m
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], nn.ReLU(inplace=False))

    replace_inplace_relu(model)
    model.eval()

    # Find last conv layer reliably
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("No Conv2d layer found in model")

    print(f"Using target layer: "
          f"{[n for n, m in model.named_modules() if m is target_layer][0]}")

    # Prepare input — requires grad for GradCAM
    input_tensor = x[sample_idx].unsqueeze(0).requires_grad_(True)

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_val)]

    # GradCAM handles grad context internally — do NOT wrap in torch.no_grad()
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Check for degenerate output
    if np.isnan(grayscale_cam).any() or grayscale_cam.max() == 0:
        print("WARNING: GradCAM produced zero or NaN output — "
              "gradients may still be vanishing. "
              "Check that the model produces non-zero output for this input.")

    # Denormalise image
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    rgb_img = x[sample_idx].cpu().numpy()
    rgb_img = np.clip(std * rgb_img + mean, 0, 1)
    rgb_img = np.transpose(rgb_img, (1, 2, 0))

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM++ | Pred: {pred_val}", fontsize=14, fontweight='bold')

    axes[0].imshow(rgb_img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title("Grad-CAM++ Heatmap")
    axes[1].axis('off')

    axes[2].imshow(cam_image)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("PAF-output", exist_ok=True)
    plt.savefig(f"PAF-output/gradcam_{sample_idx}.png")
    plt.show()
    return fig

def gradcam_visualization_old(model, x,pred_val):
    from captum.attr import LayerGradCam
    import torch.nn.functional as F
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
  

    # 1. Initialize LayerGradCam targeting your specific layer
    # model.layer4[1].conv2 is the standard attribute path for layer4_1_conv2
    
    model.eval()  # Ensure model is in eval mode
    #show_original(x[sample_idx].unsqueeze(0))
    #show_image_inline(x[sample_idx], label=pred_val)
    
    target_layers=None
    recursive_layers = list(model.modules())
    for layer in reversed(recursive_layers):
        if isinstance(layer, torch.nn.Conv2d):
            target_layers=[layer]

    if target_layers is None:
        if hasattr(model, 'features'):
            target_layers = [model.features[-1]] 
        else:
            target_layers = [model.layer4[-1]]    # Create the GradCAM object
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Prepare your input image (batch of 1)
    input_tensor = x[sample_idx].unsqueeze(0)   # shape: (1, 3, 224, 224)

    # Choose the target class (your predicted class)
    targets = [ClassifierOutputTarget(pred_val)]

    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Prepare the ORIGINAL RGB image for overlay
    rgb_img = x[sample_idx].cpu().numpy()       # shape: (3, H, W)

    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    rgb_img = std * rgb_img + mean                   # Now shapes match
    rgb_img = np.clip(rgb_img, 0, 1)                 # Clamp to [0, 1]
    rgb_img = np.transpose(rgb_img, (1, 2, 0))       # Convert to (H, W, C) for display
    #if rgb_img.max() > 1.0:                          # If values are in [0, 255]
       # rgb_img = rgb_img.astype(np.float32) / 255.0

    # Create the overlay
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM++ | Pred: {pred_val}", fontsize=14, fontweight="bold")

    axes[0].imshow(rgb_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cam_image)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.title("GRADCAM")
    plt.savefig("PAF-output/gradcam_" + ".png")

def contemporary_tools(model, input_tensor):
    if not comparative_analysis:
        print("✓ Skipping contemporary tool comparison as comparative_analysis=False.")
        return  
    
    gradcam_visualization(model, input_tensor,0)
    
    '''
    lgc = LayerGradCam(model, target_layer)

    # 2. Generate Attribution
    # target=0 is the Tench class
    attribution = lgc.attribute(input_tensor, target=0)

    # 3. Upsample to match your image size (224x224)
    # Grad-CAM is natively 7x7, so we must interpolate it to see it over the fish
    grad_cam_map = F.interpolate(attribution, size=(224, 224), mode='bilinear', align_corners=False)

    # 4. Clean up for plotting
    grad_cam_map = grad_cam_map.squeeze().cpu().detach().numpy()
    
    # 3. LRP (Relevance flow)
    lrp_heatmap = get_refined_lrp(model, input_tensor, target_class=0)
    img_display = tensor_to_display(input_tensor)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(lrp_heatmap, cmap='hot')
    plt.title("LRP Refined (Semantic Shape)")

    plt.subplot(1, 2, 2)
    # Overlay contours at 30% and 60% like your PAF plot
    plt.imshow(img_display) # Your original RGB image
    plt.contour(lrp_heatmap, levels=[0.3, 0.6], colors=['cyan', 'yellow'], linewidths=2)
    plt.title("LRP Salient Contours")
    plt.show()
    '''
def show_original(input_tensor):
    # 1. Remove the batch dimension [1, 3, 224, 224] -> [3, 224, 224]
    img = input_tensor.squeeze(0).cpu().detach()
    
    # 2. Un-normalize (ImageNet constants)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert [C, H, W] to [H, W, C] for Matplotlib
    img = img.numpy().transpose((1, 2, 0))
    
    # Apply the reverse math: (pixel * std) + mean
    img = std * img + mean
    
    # 3. Clip to ensure values stay between 0 and 1
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    plt.title("Original Input Image")
    plt.axis('off')
    plt.show()

def show_raw_pixels(x_orig, title="Raw CIFAR Pixels"):
    # 1. Convert [3, 32, 32] -> [32, 32, 3]
    img = x_orig.permute(1, 2, 0).cpu().numpy()
    
    # 2. Normalize to [0, 1] for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(4, 4))
    
    # 3. CRITICAL: interpolation='none' or 'nearest' stops the blur
    plt.imshow(img, interpolation='none')
    
    plt.title(title)
    plt.axis('off') # Hide the coordinate numbers
    plt.show()

def show_image_inline(x_orig, label=None, pred=None):
    """
    Displays a single 32x32 image.
    x_orig: Tensor of shape [3, 32, 32]
    """
    # 1. Convert [C, H, W] -> [H, W, C]
    img = x_orig.permute(1, 2, 0).cpu().numpy()
    
    # 2. Normalize 0-1 for display (if not already)
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(3, 3))
    
    # 3. Use 'nearest' to keep the pixels sharp (not blurry)
    plt.imshow(img, interpolation='nearest')
    
    title = ""
    if label is not None: title += f"True: {label} "
    if pred is not None: title += f"| Pred: {pred}"
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_paf(x, paf_visualizer, predicted,true_label,sample_id=0,mode="cprediction"):
    x_orig=x
    
    # 1. Distribution Visualization accross layers
    viz = paf_visualizer.visualize_distributions() 
    viz.render('PAF-output/distributions_' + model_name + '_' + dataset_name+'_'+mode+str(sample_id), format='png', cleanup=True)
    
    # Heatmap and Overlay Decision Logic
    paf_visualizer.visualize_probability_decision_logic_prev(
        x=x,
        x_orig=x_orig,
        predicted=predicted,
        sample_idx=sample_idx,
        save_path="PAF-output/heatmap_" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png",
    )

    paf_visualizer.plot_signed_explanation(
        original_img=x,
        predicted=predicted,
        sample_idx=sample_idx,
        save_path="PAF-output/heatmap_signed_" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png"
        )
    paf_visualizer.plot_layer_sensitivity(save_path="PAF-output/layer_sensitivity_" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png")
    paf_visualizer.plot_paf_branching_logic(is_signed=False,save_path="PAF-output/paf_branching_" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png")
    paf_visualizer.visualize_paf_prev(
        x=x,
        x_orig=x_orig,
        predicted=predicted,
        true_label=true_label,
        save_path="PAF-output/paf_x" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png",
    )
    paf_visualizer.show_salient_features(x_orig, save_path="PAF-output/salient_" + model_name + "_" + dataset_name + "_"+mode + str(sample_id) + ".png")
    paf_visualizer.plot_salient_contours(x_orig, save_path="PAF-output/contours_" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png")
    paf_visualizer.show_clean_contours(x_orig, save_path="PAF-output/clean_contours_" + model_name + "_" + dataset_name + "_" + mode + str(sample_id) + ".png")
    print(f"✓ ImageNet Sample {sample_id} analyzed. Prediction: {predicted}, Actual: {true_label}")
    paf_visualizer.visualize_layer_results(x_orig, save_path="PAF-output/" + model_name + "_" + dataset_name + "_" + mode + str(sample_id))
    return paf_visualizer

def run_and_visualize_paf(x, model, predicted,true_label,device,sample_id):

    # Run the Attribution logic
    if not analyze_misclassification:
        #paf = run_paf_on_sample(x=x,model=model,target_label=true_label,lambda_=1, delta=1, gamma=1, eps=1e-2, output_mode="target", device=device)
        paf=PAF(model=model,debug_level=debug_level,x=x,target_class=true_label,true_class=true_label)
        paf_visualizer = PAFVisualizer(paf,misclassification=analyze_misclassification,contrastive_explanation=contrastive_interpretation,true_class=true_label,target_class=predicted)
        visualize_paf(x, paf_visualizer, predicted,true_label,sample_id,mode="cprediction")
    else:
        if not contrastive_interpretation:
            #Option 1: Run PAF separately for true and predicted classes to see the difference in attribution maps
            paf_true=PAF(debug_level=debug_level,model=model,x=x,target_class=true_label,true_class=true_label)
            paf_predicted=PAF(debug_level=debug_level,model=model,x=x,target_class=predicted,true_class=true_label)
            #paf_true = run_paf_on_sample(x=x,model=model,target_label=true_label,lambda_=1, delta=1, gamma=1, eps=1e-2, output_mode="target", device=device)
            #paf_predicted = run_paf_on_sample(x=x,model=model,target_label=predicted,lambda_=1, delta=1, gamma=1, eps=1e-2, output_mode="target", device=device)

            paf_visualizer_pred = PAFVisualizer(paf_predicted,misclassification=analyze_misclassification,contrastive_explanation=contrastive_interpretation,true_class=true_label,target_class=predicted)
            paf_visualizer_true = PAFVisualizer(paf_true,misclassification=analyze_misclassification,contrastive_explanation=contrastive_interpretation,true_class=true_label,target_class=true_label)
            paf_visualizer_pred.paf_shared = paf_true
            paf_visualizer_true.paf_shared = paf_predicted

            visualize_paf(x, paf_visualizer_true, predicted,true_label,sample_id,mode="true-classification")
            visualize_paf(x, paf_visualizer_pred, predicted,true_label,sample_id,mode="misclassification")
        
            #visualize_paf(x, paf_visualizer_true, predicted,true_label,sample_id,mode="cprediction")
            #visualize_paf(x, paf_visualizer_pred, predicted,true_label,sample_id,mode="wprediction")
        else:
        #Option 2: Run PAF once with both true and predicted labels to get a contrastive attribution map that highlights differences
            paf_true=PAF(debug_level=debug_level,model=model,output_mode  = "contrastive_explanation_true",x=x,target_class=true_label,true_class=true_label)
            paf_predicted=PAF(debug_level=debug_level,model=model,output_mode  = "contrastive_explanation_predicted",x=x,target_class=predicted,true_class=true_label)
        
            #paf_predicted = run_paf_on_sample(x=x,model=model,target_label=predicted,true_label=true_label,lambda_=1, delta=1, gamma=1, eps=1e-2, output_mode="contrastive_explanation_predicted", device=device)
            #paf_true = run_paf_on_sample(x=x,model=model,target_label=predicted,true_label=true_label,lambda_=1, delta=1, gamma=1, eps=1e-2, output_mode="contrastive_explanation_true", device=device)
            
            paf_visualizer_pred = PAFVisualizer(paf_predicted,misclassification=analyze_misclassification,contrastive_explanation=contrastive_interpretation,true_class=true_label,target_class=predicted)
            paf_visualizer_true = PAFVisualizer(paf_true,misclassification=analyze_misclassification,contrastive_explanation=contrastive_interpretation,true_class=true_label,target_class=true_label)
            
            paf_visualizer_pred.paf_shared = paf_true
            paf_visualizer_true.paf_shared = paf_predicted
            visualize_paf(x, paf_visualizer_pred, predicted,true_label,sample_id,mode="misclassification")
            visualize_paf(x, paf_visualizer_true, predicted,true_label,sample_id,mode="true-classification")

            paf_visualizer_pred.plot_diagnostic_comparison(paf_predicted.distributions["x"][sample_idx],paf_true.distributions["x"][sample_idx],x)
            paf_visualizer_pred.plot_channel_importance(paf_predicted.distributions["x"][sample_idx],paf_true.distributions["x"][sample_idx],"input")

            diff_heatmap = paf_visualizer_pred.heatmap - paf_visualizer_true.heatmap
            #diff_heatmap = np.log(diff_heatmap + 1e-8)
            plt.imshow(diff_heatmap, cmap='bwr')
            plt.colorbar()
            plt.title("Contrastive PAF Difference (Predicted - True)")
            plt.savefig("PAF-output/contrastive_difference_" + model_name + "_" + dataset_name + "_sample" + str(sample_id) + ".png")
    
    return
    

def choose_sample_and_generate_visual_interpretation(test_loader, model, device):
    test_set=enumerate(test_loader)
    for i, (x, y) in test_set:
        x = x.to(device)
        true_label = int(y.item())

        with torch.no_grad():
            logits = model(x)
        predicted = int(logits.argmax(dim=-1).item())
        predicted_is_true=predicted == y.item()
        if  predicted_is_true and not analyze_misclassification:
            print(f"✓ Sample {i} correctly predicted as {predicted}. ")
            run_and_visualize_paf(x, model, predicted, true_label,device,i)
            print(f"Running Baseline approaches...")
            contemporary_tools(model, x)
            return
        elif not predicted_is_true and analyze_misclassification:
            print(f"✓ Sample {i} will be analyzed as incorrectly predicted as {predicted} true: {true_label}. Skipping...")
            run_and_visualize_paf(x, model, predicted, true_label, device,i)
            print(f"Running Baseline approaches...")
            contemporary_tools(model, x)
            return
        else: 
            print(f"✓ Sample {i} will be skipped as Predicted: {predicted}, True Label: {true_label},  Requested misclassification analysis: {analyze_misclassification}")    
            continue
        #show_image_inline(x.cpu(), label=true_label, pred=predicted)
    print(f"No visualization sample found with misclassification ={analyze_misclassification}.")
    return 
'''
def run_randomization_tests_old(test_loader, model, device):
    all_results = {}  # name -> {'del': list of lists, 'ins': list of lists}
    method_names = ["PAF", "GCAM", "IG", "LRP","DEEPSHAP"]
    visualize_heatmap=test_runs<5
    collected = 0
    sample_id=0
    if analyze_multimode:
        rand_test=RandomizationTestMultiMode(model=model,device=device,analyze_misclassification=analyze_misclassification,contrastive_interpretation=contrastive_interpretation,sample_idx=sample_idx,patch_size=patch_size,visualize_heatmap=visualize_heatmap)
    else:
        rand_test=RandomizationTest(model=model,device=device,analyze_misclassification=analyze_misclassification,contrastive_interpretation=contrastive_interpretation,sample_idx=sample_idx,patch_size=patch_size,visualize_heatmap=visualize_heatmap)
    all_samples=[]
    while collected < test_runs:
        test_sample=get_test_sample(test_loader, model, device,analyze_misclassification,random_sample,samples_checked=sample_id) 
        idx, x, y, predicted, sample_id = test_sample
        true_label = y.item() if isinstance(y, torch.Tensor) else y
        all_samples.append({"idx":idx,"x":x,"y":true_label,"predicted":predicted, "sample_id":sample_id})
        collected += 1
    rand_test.run(all_samples,labels,"PAF-output/final_scores.png")
'''

def run_randomization_tests(test_loader, model, device,model_name,hook_manager):
    all_results = {}  
    method_names = ["PAF", "GCAM", "IG", "LRP","DEEPSHAP"]
    visualize_heatmap=test_runs<5
    collected = 0
    sample_id=0
    rand_test=RandomizationTestMultiMode(
        model=model,
        device=device,
        analyze_misclassification=analyze_misclassification,
        contrastive_interpretation=contrastive_interpretation,
        sample_idx=sample_idx,
        patch_size=patch_size,
        visualize_heatmap=visualize_heatmap,
        paf_modes=paf_modes,
        model_name=model_name,
        hook_manager=hook_manager
        )
    all_samples=[]
    while collected < test_runs:
        test_sample=get_test_sample(test_loader, model, device,analyze_misclassification,random_sample,samples_checked=sample_id) 
        idx, x, y, predicted, sample_id = test_sample
        true_label = y.item() if isinstance(y, torch.Tensor) else y
        all_samples.append({"idx":idx,"x":x,"y":true_label,"predicted":predicted, "sample_id":sample_id})
        collected += 1
    results=rand_test.run(all_samples,labels,"PAF-output/"+model_name+".png",test_type='both')
    rand_test.cleanup
    del rand_test
    del test_sample
    return results


def run_multi_score_paf(test_loader, model, device):
    test_sample=get_test_sample(test_loader, model, device,analyze_misclassification,random_sample) 
    idx, x, y, predicted, sample_id = test_sample
    true_label = y.item() if isinstance(y, torch.Tensor) else y
    paf=PAF(
        model=model,
        debug_level=debug_level,
        x=x,
        target_class=true_label,true_class=true_label,
        modes       = paf_modes
        )
    visualizer = PAFVisualizer(paf, true_class=true_label, target_class=predicted)
    visualizer.visualize_heatmap_all_mode(
        x=x,
        sample_id=sample_id,
        save_path="PAF-output/heatmaps_" + model_name + "_" + dataset_name + "_" + str(sample_id) + ".png"
        )
    save_path="PAF-output/distributions_"+ model_name + "_" + dataset_name + "_"+ str(sample_id) + ".png"
    visualizer.plot_paf_layerwise_distribution(save_path=save_path)
    paf.cleanup
    del paf, visualizer
    
    #contemporary_tools(model, x)
    #run_randomization_tests(test_loader, model, device)
    #run_perturbation_tests(test_loader,model,device,test_runs)

def get_test_sample(test_loader, model, device,misclassification=False,random_sample=False,samples_checked=0):
    test_set=test_loader.dataset
    num_samples = len(test_set)
    i = random.randint(0, num_samples - 1) if random_sample else 0

    while samples_checked < num_samples:
        i = random.randint(0, num_samples - 1) if random_sample else 0
        idx = (i + samples_checked) % num_samples
        x, y = test_set[idx]
        x=x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        predicted = int(logits.argmax(dim=-1).item())
        true_label = y.item() if isinstance(y, torch.Tensor) else y
        is_correct = (predicted == true_label)
        matches_criteria = (is_correct != misclassification)
        if matches_criteria:
            # 3. Successful match: Run hooks and return
            #hook_manager.run_forward(x)
            return (idx, x, y,predicted,samples_checked)
        samples_checked += 1
    raise RuntimeError("No matching sample found in dataset")

    
def run_purturbation_tests_all_model():
    loader = ModelConfigLoader("models/models_config.yaml")
    factory = ModelFactory(loader)
    #models=['resnet18','resnet34','resnet50','vgg16','vgg19','vit_b_16']
    models=['resnet18']
    global_metrics = {}
    steps_shared = None
    for mod_name in models:
        print(f"\n{'='*20}\nEvaluating Model: {mod_name}\n{'='*20}")
        test_config = TrainingConfig(
        model=mod_name,  
        models_config_path=yaml_config_path,
        num_classes=1000, #for real imagenet, 100 for imagenet-100
        dataset=dataset_name,    #check the dataset name in models_config.yaml, it should match the one there
        data_path=dataset_path,
        #dataset="imagenet",
        batch_size=1,
        device=str(device),
        num_workers=4
        )
        model = factory.create_model(
            model_name=test_config.model,
            num_classes=test_config.num_classes,
            pretrained=True
            ).to(device)
        model.eval()
        hook_manager=PAFHookManager(model)
        test_loader = factory.get_dataloader(model_name=test_config.model,subset="val", config=test_config)
        steps, stats, model_results = run_perturbation_tests(test_loader,model,device,test_runs,mod_name,hook_manager)
        
        if steps_shared is None: steps_shared = steps

        for method, res in model_results.items():
            if method not in global_metrics:
                global_metrics[method] = {'ins': [], 'del': []}
            
            # Convert list of sample curves to a numpy array for this model
            # res['ins'] is a list of lists (samples x steps)
            global_metrics[method]['ins'].append(np.array(res['ins']))
            global_metrics[method]['del'].append(np.array(res['del']))
    
    # Generate the aggregate visualization
    plot_neurips_aggregate(steps_shared, global_metrics)
    
    # Print the numeric data for your LaTeX table
    #generate_summary_latex_data(steps_shared, global_metrics)
    generate_summary_table(global_metrics)

def run_perturbation_tests(
    test_loader,
    model,
    device,
    hook_manager,
    n_samples   :int  = 100,
    mod_name  :str=""
):
    visualize_heatmap = n_samples < 5
    collected         = 0
    sample_id         = 0

    perturb_test = PerturbationTestMultiMode(
        model                      = model,
        model_name                 = mod_name,
        hook_manager               = hook_manager,
        analyze_misclassification  = analyze_misclassification,
        contrastive_interpretation = contrastive_interpretation,
        sample_idx                 = sample_idx,
        patch_size                 = patch_size,
        visualize_heatmap          = visualize_heatmap,
        paf_modes                  = paf_modes,
    )

    # Build method names dynamically from what PerturbationTest will produce
    baseline_names = ["GCAM", "IG", "LRP", "DEEPSHAP"]
    paf_names      = perturb_test.paf_method_names
    method_names   = baseline_names + paf_names

    all_results = {name: {'del': [], 'ins': []} for name in method_names}

    while collected < n_samples:
        test_sample = get_test_sample(
            test_loader, model, device,
            analyze_misclassification,
            random_sample,
            samples_checked=sample_id
        )
        idx, x, y, predicted, sample_id = test_sample
        true_label = y.item() if isinstance(y, torch.Tensor) else y

        try:
            steps, results = perturb_test.run_perturbation_tests(
                x, true_label, predicted, device, idx=idx
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
                perturb_test.plot_faithfulness_single_test(steps, results, idx)

            print(f"[{collected}/{n_samples}] sample {idx} done")

        except Exception as e:
            print(f"Sample {idx} failed: {e} — skipping")
            sample_id += 1
            continue

    # Filter to methods that have at least one result
    valid_methods = [
        n for n in method_names
        if len(all_results.get(n, {}).get('del', [])) > 0
    ]
    if not valid_methods:
        print("No valid results collected.")
        return None, None, None

    # Compute statistics
    stats, audc, auic, del_matrix, ins_matrix = \
        perturb_test.compute_aggregate_statistics(
            all_results, valid_methods, steps
        )

    # Plot and print
    perturb_test.plot_aggregated_results(steps, del_matrix, ins_matrix, valid_methods)
    perturb_test.print_summary_table(stats, valid_methods,save_path="PAF-output/aggregated_results"+mod_name+".txt")

    # Save raw matrices
    os.makedirs("PAF-output", exist_ok=True)
    np.save("PAF-output/raw_del.npy", {n: del_matrix[n] for n in valid_methods})
    np.save("PAF-output/raw_ins.npy", {n: ins_matrix[n] for n in valid_methods})

    return steps, stats, all_results

def main():

    # 1. Initialize Factory and Config
    loader = ModelConfigLoader("models/models_config.yaml")
    factory = ModelFactory(loader)

    # data_path contains 'val' and 'train' subdirectories.
    test_config = TrainingConfig(
        model=model_name,  
        models_config_path=yaml_config_path,
        num_classes=1000, #for real imagenet, 100 for imagenet-100
        dataset=dataset_name,    #check the dataset name in models_config.yaml, it should match the one there
        data_path=dataset_path,
        #dataset="imagenet",
        batch_size=1,
        device=str(device),
        num_workers=4
    )

    # 2. Setup Model (Pretrained ResNet-18)
    model = factory.create_model(
        model_name=test_config.model,
        num_classes=test_config.num_classes,
        pretrained=True
    ).to(device)
    model.eval()

    # 3. Setup Hooks and Trace Graph
    #hook_manager = PAFHookManager(model)
    
    # 4. Load ImageNet Val Set using the Factory's new method
    # This automatically applies the 256->224 resize/crop and ImageNet normalization
    print("Loading Validation samples...")

    # Only for CIFAR-10/100, for ImageNet we use the dataloader directly to get the original images
    '''
    test_loader = factory.get_torchvision_loader(
        dataset_name=test_config.dataset,
        config=test_config,
       is_train=False
    )
    '''
    test_loader = factory.get_dataloader(model_name=test_config.model,subset="val", config=test_config)
    #choose_sample_and_generate_visual_interpretation(test_loader, model, device)
    #run_purturbation_tests(test_loader, model, device,test_runs)
    #run_randomization_tests(test_loader, model, device)
    run_multi_score_paf(test_loader, model, device)
    #run_single_test(hook_manager, test_loader, model, device)

def run_randomization_tests_all_model():
    loader = ModelConfigLoader("models/models_config.yaml")
    factory = ModelFactory(loader)
    #models=['resnet18','resnet34','resnet50','vgg16','vgg19','vit_b_16']
    models=['vgg16']
    global_metrics = {}
    steps_shared = None
    results={}
    for mod_name in models:
        print(f"\n{'='*20}\nEvaluating Model: {mod_name}\n{'='*20}")
        test_config = TrainingConfig(
            model=mod_name,  
            models_config_path=yaml_config_path,
            num_classes=1000, #for real imagenet, 100 for imagenet-100
            dataset=dataset_name,    #check the dataset name in models_config.yaml, it should match the one there
            data_path=dataset_path,
            batch_size=1,
            device=str(device),
            num_workers=4
            )
        model = factory.create_model(
            model_name=test_config.model,
            num_classes=test_config.num_classes,
            pretrained=True
            ).to(device)
        model.eval()
        hook_manager = PAFHookManager(model)
        test_loader = factory.get_dataloader(model_name=test_config.model,subset="val", config=test_config)
        results[mod_name]=run_randomization_tests(test_loader,model,device,mod_name,hook_manager)
    
        rows = []
        for tt, entries in results[mod_name].items():
            for entry in entries:
                row = entry.copy()
                row['test_type'] = tt
                row['mod_name'] = mod_name
                rows.append(row)
        
        # Save this model's specific data
        df_temp = pd.DataFrame(rows)
        os.makedirs("results_checkpoints", exist_ok=True)
        df_temp.to_csv(f"results_checkpoints/{mod_name}_results.csv", index=False)
        print(f"Checkpointed results for {mod_name}")

    # Initialize an empty list to collect rows
    '''
    rows = []

    # Iterate through each model in the results
    for mod_name, tests in results.items():
        # Iterate through each test type (weight, activation)
        for tt, entries in tests.items():
            # Iterate through the list of sample results
            for entry in entries:
                # Create a shallow copy to avoid modifying original data
                row = entry.copy()
                
                # Add the test type as a column for easy filtering later
                row['test_type'] = tt 
                
                # Ensure model name is explicitly in the row if not already
                row['mod_name'] = mod_name 
                
                rows.append(row)

    # Create the final DataFrame
    df = pd.DataFrame(rows)
    '''
    df=load_and_plot_all()
    # Clean up: Ensure numeric types for metrics
    df['spearman'] = pd.to_numeric(df['spearman'], errors='coerce')
    df['ssim'] = pd.to_numeric(df['ssim'], errors='coerce')
    weight_df = df[df['test_type'] == 'weight']
    activation_df = df[df['test_type'] == 'activation']
    plot_specific_randomization(weight_df, test_type="weight")
    plot_specific_randomization(activation_df, test_type="activation")

import glob

def load_and_plot_all():
    # 1. Gather all CSV files in the checkpoint directory
    path = "results_checkpoints"
    all_files = glob.glob(os.path.join(path, "*.csv"))

    if not all_files:
        print("No results found!")
        return

    # 2. Combine them into one big DataFrame
    df_list = [pd.read_csv(f) for f in all_files]
    final_df = pd.concat(df_list, ignore_index=True)
    
    print(f"Loaded results for {final_df['mod_name'].nunique()} models.")
    
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
    g.fig.suptitle(f"Sanity Check: {title_suffix}", fontweight='bold', y=1.05)
    
    for ax in g.axes.flat:
        ax.axhline(0.3, ls='--', color='red', alpha=0.6, label='Threshold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    g.add_legend()
    plt.savefig(f"PAF-output/{test_type}_randomization_final_results.pdf", bbox_inches='tight')
    plt.show()   

if __name__ == "__main__":
    #main()
    #run_purturbation_tests_all_model()
    run_randomization_tests_all_model()
    