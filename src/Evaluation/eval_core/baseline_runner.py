"""
Baseline Runner
===============
Wraps GradCAM++, LRP, DeepSHAP, IG with a consistent interface.
Used by all evaluation scripts — no baseline code duplicated elsewhere.

_make_shap_safe is here only — remove from nn_graph.py and pointing_game.py.
"""

import torch
import torch.nn as nn
import numpy as np
import types
from typing import Dict, Optional
from Evaluation.eval_core.heatmap_utils import _to_numpy_hw


class BaselineRunner:
    """
    Runs all baseline attribution methods on a single sample.
    Construct once, call run() per sample.

    Args:
        model:        nn.Module in eval mode
        target_layer: nn.Module — layer for GradCAM/LRP/IG/DeepSHAP
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model        = model
        self.target_layer = target_layer
        self._shap_model  = None   # lazy — built on first DeepSHAP call

    def run(
        self,
        x:      torch.Tensor,
        label:  int,
        H:      int,
        W:      int,
        methods: Optional[list] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Run all baselines. Returns {method_name: heatmap_np or None}.

        Args:
            methods: subset to run, e.g. ['GradCAM++', 'LRP'].
                     None = run all.
        """
        all_methods = methods or ['GradCAM++', 'LRP', 'DeepSHAP', 'IG']
        results = {}

        for method in all_methods:
            try:
                results[method] = self._run_one(method, x, label, H, W)
            except Exception as e:
                print(f"{method} failed: {e}")
                results[method] = None

        return results

    def _run_one(self, method, x, label, H, W):
        # Find target layer if not provided
        target_layer = self._get_target_layer()
        original_device = next(self.model.parameters()).device
        use_cpu = original_device.type == 'mps'

        if method == 'GradCAM++':
            from pytorch_grad_cam import GradCAMPlusPlus
            cam = GradCAMPlusPlus(model=self.model,
                                  target_layers=[target_layer])
            h   = cam(input_tensor=x)[0]
            h   = np.clip(h, 0, None)
            return h / h.max() if h.max() > 0 else h

        # For all captum methods: move to CPU if on MPS
        target_layer_name = next(
                (n for n, m in self.model.named_modules() if m is target_layer),
                None
            )
        if target_layer_name is None:
            raise ValueError("target_layer not found in model")
        if use_cpu:
            self.model.cpu()
            x_run = x.cpu()
            # Find target layer name, then look it up in CPU model
        else:
            x_run = x
        
        target_layer_run = dict(self.model.named_modules()).get(target_layer_name)
        if target_layer_run is None:
            raise ValueError(f"Layer '{target_layer_name}' not found after device move")
        
        result=None
        try:    
            if method == 'LRP':
                from captum.attr import LayerLRP
                model_name = getattr(self.model, 'name', '').lower()
                if 'vit' in model_name:
                    raise ValueError("LRP not supported for ViT")
                lrp    = LayerLRP(self.model, target_layer_run)
                attr = lrp.attribute(x_run, target=label)
                result=_to_numpy_hw(attr, H, W)
                del lrp
                
            elif method == 'DeepSHAP':
                from captum.attr import LayerDeepLiftShap
                shap_model  = self._get_shap_model()
                if use_cpu:
                    shap_model = shap_model.cpu()
                shap_layer  = _find_layer_in_copy(self.model, shap_model, target_layer_run)                
                baseline = torch.zeros(5, *x_run.shape[1:], device=x_run.device)
                attr = LayerDeepLiftShap(shap_model, shap_layer)\
                        .attribute(x_run, baselines=baseline, target=label)
            
                result=_to_numpy_hw(attr, H, W)

            elif method == 'IG':
                from captum.attr import LayerIntegratedGradients
                safe_model  = self._get_shap_model()   # already built and cached
                safe_model  = safe_model.cpu() if use_cpu else safe_model
                safe_layer  = _find_layer_in_copy(self.model, safe_model, target_layer_run)
                attr = LayerIntegratedGradients(safe_model, safe_layer)\
                        .attribute(
                            x_run,
                            baselines=torch.zeros_like(x_run),
                            target=label,
                            n_steps=10,
                            internal_batch_size=1,
                        )
                result=_to_numpy_hw(attr, H, W)
            else:
                raise ValueError(f"Unknown method: {method}")

        finally:
            # Always restore model to original device
            if use_cpu:
                self.model.to(original_device)
        return result

    def _get_shap_model(self) -> nn.Module:
        """Build shap-safe model once, reuse across samples."""
        if self._shap_model is None:
            self._shap_model = _make_shap_safe(self.model)
        return self._shap_model
    
    def _get_target_layer(self) -> nn.Module:
        if self.target_layer is not None:
            return self.target_layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found in model")

# ----------------------------------------------------------------
# Shared utilities — single source, used by BaselineRunner only
# ----------------------------------------------------------------

def _make_shap_safe(model: nn.Module) -> nn.Module:
    """Non-inplace ReLU model for DeepSHAP. Uses state_dict — no deepcopy."""
    import torchvision.models as tvm
    from torchvision.models.resnet import BasicBlock, Bottleneck

    model_name = getattr(model, 'name', None)

    arch_map = {
        'resnet18': tvm.resnet18, 
        'resnet34': tvm.resnet34, 
        'resnet50': tvm.resnet50, 
        'resnet101': tvm.resnet101, 
        'vgg16': tvm.vgg16, 
        'vgg19': tvm.vgg19,
        'vit_b_16':  tvm.vit_b_16,
        'vit_b_32':  tvm.vit_b_32,
        'vit_l_16':  tvm.vit_l_16,
        'vit_l_32':  tvm.vit_l_32,
    }
    builder    = arch_map.get(model_name) 
    if builder is None:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Add it to _make_shap_safe."
        )

    shap_model = builder(weights=None)
    shap_model.load_state_dict(model.state_dict())
    shap_model.eval()

    def _replace_relus(m):
        for name, child in m.named_children() :
            if isinstance(child, nn.ReLU) and child.inplace:
                setattr(m, name, nn.ReLU(inplace=False))
            else:
                _replace_relus(child)

    _replace_relus(shap_model)

    def _bb(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)

    def _btn(self, x):
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
            m.forward = types.MethodType(_bb, m)
        elif isinstance(m, Bottleneck):
            m.relu1 = nn.ReLU(inplace=False) 
            m.relu2 = nn.ReLU(inplace=False) 
            m.relu3 = nn.ReLU(inplace=False)
            m.forward = types.MethodType(_btn, m)

    return shap_model


def _find_layer_in_copy(original, copy, target):
    for name, mod in original.named_modules():
        if mod is target:
            return dict(copy.named_modules())[name]
    raise ValueError("target layer not found in original model")
