import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
#from zennit.composites import EpsilonPlusFlat
#from zennit.canonizers import SequentialMergeBatchNorm

from captum.attr import IntegratedGradients, LRP, DeepLiftShap
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from core.compat.shap import make_model_universal_for_shap

class XAI:
    def __init__(self, model):
        self.model=model
        self.shap_model=make_model_universal_for_shap(model)
    
    def cleanup(self):
        del self.model
        del shap_model
         
    def get_deepshap_heatmap(self, input_tensor, target_class):
        """
        Computes DeepSHAP attributions for a specific class.
        """
        model = self.shap_model
        #model.eval()
        dl_shap = DeepLiftShap(model)
        
        # Create a baseline (usually zeros). 
        # Using multiple baselines makes SHAP more stable.
        baseline = torch.zeros(5, *input_tensor.shape[1:]).to(input_tensor.device)
        
        # input_tensor should be (1, C, H, W)
        attributions = dl_shap.attribute(input_tensor, 
                                        baselines=baseline, 
                                        target=target_class)
        
        # Sum across channels (C) and convert to numpy (H, W)
        heatmap = torch.sum(attributions.squeeze(0), dim=0).cpu().detach().numpy()
        
        # Optional: Take absolute values or only positive (ReLU-style)
        heatmap = np.maximum(0, heatmap) 
        del dl_shap
        return heatmap

    def get_integrated_gradient_map(self, x, target_class):
        ig = IntegratedGradients(self.model)  
        # n_steps=50 is standard for IG stability
        attr_ig, delta = ig.attribute(x, target=target_class, return_convergence_delta=True)
        heatmap_ig = attr_ig.abs().sum(dim=1).squeeze().cpu().detach().numpy()
        del ig
        return heatmap_ig

    def get_lrp_map(self, x, target_class):
        lrp = LRP(self.model)
        attr_lrp = lrp.attribute(x, target=target_class)
        heatmap_lrp = attr_lrp.sum(dim=1).squeeze().cpu().detach().numpy()
        del lrp
        return heatmap_lrp

    def get_gradcam_map(self, x_tensor, target_class):
        # 1. Ensure model is in eval and find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        # 2. Generate CAM
        # We use the specific target_class to see what GradCAM thinks is important for that label
        cam = GradCAMPlusPlus(model=self.model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(target_class)]
        
        # grayscale_cam shape is [1, H, W]
        grayscale_cam = cam(input_tensor=x_tensor, targets=targets)[0]
        del cam
        return grayscale_cam
