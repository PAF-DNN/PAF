"""
Model I/O utilities - Save and load model state dicts.

Separated from model_factory.py for cleaner separation of concerns.
"""

import torch
import torch.nn as nn
from pathlib import Path


def save_model(model: nn.Module, path: str) -> None:
    """
    Save model state dict to disk.

    Parameters
    ----------
    model : nn.Module
        Model to save
    path : str
        Path to save to (parent directories created automatically)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✓ Model saved to {path}")


def load_model(
    model: nn.Module,
    path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load model state dict from disk.

    Parameters
    ----------
    model : nn.Module
        Model instance to load state into
    path : str
        Path to load from
    device : torch.device
        Device to load onto

    Returns
    -------
    nn.Module
        Model with loaded state, in eval mode
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model loaded from {path}")
    return model
