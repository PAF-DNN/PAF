"""
PAF Utilities
============

Utility functions for PAF including parsing, naming, and key generation.
"""

from torch import nn
import types
from .scoring import ScoringMode


def parse_mode_from_string(mode_str: str) -> ScoringMode:
    """Parse scoring mode from string, handling common prefixes."""
    # Clean mode string
    clean_mode = mode_str.strip()
    
    # Remove common prefixes
    for prefix in ["PAF.", "scoring.", "paf.", "ScoringMode."]:
        if clean_mode.startswith(prefix):
            clean_mode = clean_mode[len(prefix):]
            break
    
    # Direct enum lookup (Python enums support this)
    try:
        return ScoringMode(clean_mode.lower())
    except ValueError:
        raise ValueError(f"Unknown scoring mode: {mode_str}. "
                        f"Available: {[m.value for m in ScoringMode]}")


def make_mode_name(mode: ScoringMode, **kwargs) -> str:
    """Create human-readable name for a scoring mode."""
    tau = kwargs.get('tau', 1.0)
    if mode == ScoringMode.SIGNED_SPLIT:
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 0.0)
        return f"PAF-{mode.value} τ={tau} α={alpha} β={beta}"
    return f"PAF-{mode.value} τ={tau}"


def make_mode_key(mode: ScoringMode, tau: float = 1.0, **kwargs) -> tuple:
    """Hashable key for (mode, hyperparameters) — used as dict key in PAF."""
    if mode == ScoringMode.SIGNED_SPLIT:
        return (mode, tau, kwargs.get('alpha', 1.0), kwargs.get('beta', 0.0))
    return (mode, tau)

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
