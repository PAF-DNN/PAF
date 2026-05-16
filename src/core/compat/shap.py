"""
DeepSHAP Compatibility Utilities
=================================
Functions to make models compatible with DeepSHAP attribution methods.
"""

import torch.nn as nn
import copy
import types
from torchvision.models.resnet import BasicBlock, Bottleneck


def make_model_universal_for_shap(model: nn.Module) -> nn.Module:
    """
    Creates a deepcopy of the model safe for DeepSHAP.
    Replaces shared ReLU instances with unique non-inplace ones.
    Works for ResNet BasicBlock and Bottleneck.
    """
    shap_model = copy.deepcopy(model)

    def replace_relus(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                replace_relus(child)

    replace_relus(shap_model)

    def basicblock_forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)

    def bottleneck_forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu3(out + identity)

    for module in shap_model.modules():
        if isinstance(module, BasicBlock):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.forward = types.MethodType(basicblock_forward, module)
        elif isinstance(module, Bottleneck):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.relu3 = nn.ReLU(inplace=False)
            module.forward = types.MethodType(bottleneck_forward, module)

    return shap_model
