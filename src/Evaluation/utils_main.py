import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights
from scipy import stats
import pandas as pd
from core.paf import *
from Evaluation.model_factory import ModelConfigLoader, ModelFactory, TrainingConfig
from core.nn_graph import PAFHookManager
from core.paf_visualizer import PAFVisualizer
from Evaluation.randomization_test_multimode import *
from Evaluation.perturbation_test_multimode import *
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

@staticmethod
def _paf_mode_key(mode) -> str:
        (mode, kwargs)=mode
        t   = kwargs.get('tau', 1.0)
        if mode == ScoringMode.SIGNED_SPLIT:
            alpha = kwargs.get('alpha', 1.0)
            beta  = kwargs.get('beta',  0.0)
            key = (mode, t, alpha, beta)   # unique key per alpha/beta combination
        else:
            key = (mode, t)   
        return key # unicode \u03C4

def resolve_device(device_config: str = "auto") -> torch.device:
    if not isinstance(device_config, str):
        print(f"Invalid device config {device_config!r}; falling back to auto.")
        device_config = "auto"

    device_config = device_config.lower().strip()

    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(device_config)
    except RuntimeError:
        print(f"Invalid device '{device_config}'; falling back to CPU.")
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    return device


def load_model_dataset():

    config_data=load_config("config/evaluation_config.yaml")
    model_name = config_data['model']['name']
    model_config_path=config_data['model']['model_config_path']
    dataset_name = config_data['dataset']['name']
    dataset_path= config_data['dataset']['path']
    experiment_config = config_data.get('experiment', {})
    shuffle=experiment_config.get('random_sample', False)
    batch_size = config_data['dataset']['batch_size']
    device= resolve_device(experiment_config.get('device', 'auto'))
    # 1. Initialize Factory and Config
    loader = ModelConfigLoader("config/models_config.yaml")
    factory = ModelFactory(loader)
    # data_path contains 'val' and 'train' subdirectories.
    test_config = TrainingConfig(
        model=model_name,  
        models_config_path=model_config_path,
        num_classes=1000,       #for real imagenet, 100 for imagenet-100
        dataset=dataset_name,    #check the dataset name in models_config.yaml, it should match the one there
        data_path=dataset_path,
        #dataset="imagenet",
        batch_size=batch_size,
        device=str(device),
        num_workers=4,
        shuffle=shuffle
    )

    # 2. Setup Model (Pretrained ResNet-18)
    model = factory.create_model(
        model_name=test_config.model,
        num_classes=test_config.num_classes,
        pretrained=True
    ).to(device)
    model.eval()
    hook_manager = PAFHookManager(model)
    print("Loading Validation samples...")
    test_loader = factory.get_dataloader(model_name=test_config.model,subset="val", config=test_config)
    return model, hook_manager, test_loader, config_data, device

def modify_testsample(testset,model,device):
    x, y = testset
    x=x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
    predicted = int(logits.argmax(dim=-1).item())
    true_label = y.item() if isinstance(y, torch.Tensor) else y
    return 27026,x,y,predicted,27026

#unoptimized
def get_test_sample(test_loader, model, device,misclassification=False,random_sample=True,samples_checked=0):
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

def find_a_random_image():
    
    import os
    from torchvision.utils import save_image
    model, hook_manager, test_loader, config_data=load_model_dataset()

    # The official ImageNet label is "drilling platform" or "offshore rig"
    target_label = "drilling platform" 

    test_set = test_loader.dataset
    '''
    found_count = 0

    # Create a temp folder to look at the results
    os.makedirs("found_rigs", exist_ok=True)

    for i in range(len(test_set)):
        img_tensor, label_idx = test_set[i]
        class_name = labels[label_idx]
        print(f"ID: {i}, class name: {class_name}")
        if target_label in class_name.lower():
            print(f"Match found! ID: {i}, Label: {class_name}")
            
            # Save the image so you can check if it's the right one
            save_image(img_tensor, f"found_rigs/rig_{i}.png")
            
            found_count += 1
            if found_count > 10: # Stop after 10 so you don't save all 50
                break
                
    '''
    # Based on your output, the block starts around here
    start_id = 27000 
    end_id = 27200
    save_dir = "check_rigs"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Dumping images from {start_id} to {end_id}...")

    for i in range(start_id, end_id):
        img_tensor, label_idx = test_loader.dataset[i]
        # Save with index so you know exactly which ID to hardcode later
        save_image(img_tensor, f"{save_dir}/id_{i}.jpg")

def get_test_samples(test_loader, model, device, misclassification=False, num_needed=1):
    """
    Optimized batch-based sampler to find valid test samples.
    """
    model.eval()
    found_samples = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 1. Batch Inference (Highly Optimized)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            
            # 2. Vectorized Criteria Check
            # If misclassification=True, we look for (preds != labels)
            if misclassification:
                mask = (preds != labels)
            else:
                mask = (preds == labels)
            
            # 3. Extract matching indices
            indices = mask.nonzero(as_tuple=True)[0]
            
            for idx in indices:
                # Store (image, true_label, predicted_label)
                found_samples.append((
                    images[idx].unsqueeze(0), 
                    labels[idx].item(), 
                    preds[idx].item()
                ))
                
                if len(found_samples) >= num_needed:
                    return found_samples
                    
    raise RuntimeError(f"Could not find {num_needed} samples matching criteria.")

def get_canny_overlay(image_tensor, low_threshold=100, high_threshold=200):
    """
    image_tensor: [3, H, W] torch tensor (normalized 0-1)
    """
    # 1. Convert tensor to grayscale CV2 image
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Apply Canny
    # These thresholds determine how many 'edges' you see. 
    # For XAI, you want only the main structural edges.
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges

def plot_paf_with_edges(heatmap, edges):
    # Normalize heatmap for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    plt.imshow(heatmap, cmap='jet')
    # Overlay edges: use a binary alpha mask so only the white lines show
    plt.imshow(edges, cmap='gray', alpha=0.2) 
    plt.axis('off')
    plt.show()
