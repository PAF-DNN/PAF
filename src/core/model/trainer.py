"""
Training Utilities - Optimizer, epoch training, evaluation.

Separated from model_factory.py for cleaner architecture.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from core.model.config import TrainingConfig


def get_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """
    Create optimizer from training config.

    Parameters
    ----------
    model : nn.Module
        Model to optimize
    cfg : TrainingConfig
        Training configuration with optimizer type, learning rate, etc.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer instance
    """
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train
    loader : DataLoader
        Training dataloader
    optimizer : torch.optim.Optimizer
        Optimizer instance
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to train on

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on dataset.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    loader : DataLoader
        Evaluation dataloader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to evaluate on

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total
