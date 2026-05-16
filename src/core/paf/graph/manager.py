"""
PAF Graph Manager
==================
Coordinates graph tracing (once) and activation capture (per forward pass).

Responsibilities:
  - Own GraphInfo (immutable, created once)
  - Own ActivationStore (mutable, updated per forward pass)
  - Refine node_types after first forward pass
  - Expose clean interface to PAF
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from core.paf.graph.tracer import GraphInfo, trace_model
from core.paf.graph.activations.store import ActivationStore


class PAFGraphManager:
    """
    Manages PAF graph structure and activation capture.

    Graph tracing happens ONCE at construction.
    Activation capture happens on each call to run_forward().

    Args:
        model: nn.Module in eval mode

    Usage:
        manager = PAFGraphManager(model)
        manager.run_forward(x)

        # Access graph structure
        graph_info = manager.graph_info

        # Access activations
        activations = manager.activations
        act = manager.activations.get('conv1')
    """

    def __init__(self, model: nn.Module):
        model.eval()
        self._model = model

        # Trace ONCE — immutable from here
        self._graph_info     = trace_model(model)

        # Activation store — updated per forward pass
        self._activation_store = ActivationStore(self._graph_info.traced)

        # Track whether first forward pass has run
        # Node types are refined after first pass when activation shapes known
        self._node_types_refined = False

    # ----------------------------------------------------------------
    # Primary interface
    # ----------------------------------------------------------------

    def run_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass and capture activations.
        Graph structure is not re-traced.
        """
        output = self._activation_store.run(x)

        # Refine node types once — after first forward pass only
        if not self._node_types_refined:
            self._graph_info.update_node_types(
                activations = self._activation_store.activations,
                module_map  = self._graph_info.module_map,
            )
            self._node_types_refined = True

        return output

    # ----------------------------------------------------------------
    # Read-only accessors
    # ----------------------------------------------------------------

    @property
    def graph_info(self) -> GraphInfo:
        """Immutable graph structure — safe to cache."""
        return self._graph_info

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        """Current activations — changes after each run_forward()."""
        return self._activation_store.activations

    @property
    def model(self) -> nn.Module:
        return self._model

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def retrace(self) -> None:
        """
        Force re-trace of model. Only needed if model architecture changed.
        Resets node type refinement.
        """
        self._graph_info       = trace_model(self._model)
        self._activation_store = ActivationStore(self._graph_info.traced)
        self._node_types_refined = False

    def clear_activations(self) -> None:
        self._activation_store.clear()

    def diagnose(self, x: torch.Tensor) -> None:
        """Print activation capture status for all nodes."""
        self.run_forward(x)
        graph_info = self._graph_info

        print(f"\n{'Status':<6} {'Node Name':<45} {'Type':<15} {'Shape'}")
        print("-" * 90)

        missing = 0
        for node in graph_info.traced.graph.nodes:
            if node.op == 'output':
                continue
            name      = node.name
            ntype     = graph_info.node_types.get(name, 'unknown')
            act       = self._activation_store.get(name)
            shape_str = str(tuple(act.shape)) if act is not None else 'MISSING'
            status    = '✓' if act is not None else '✗'
            if act is None:
                missing += 1
            print(f"{status:<6} {name:<45} {ntype:<15} {shape_str}")

        print(f"\nTotal: {len(self._activation_store)} captured, {missing} missing")

    def __repr__(self) -> str:
        return (
            f"PAFGraphManager("
            f"model={type(self._model).__name__}, "
            f"graph={self._graph_info}, "
            f"activations={len(self._activation_store)} nodes"
            f")"
        )


# ----------------------------------------------------------------
# Convenience loader — unchanged interface
# ----------------------------------------------------------------

def load_model_with_manager(
    model_name: str  = 'resnet18',
    pretrained:  bool = True,
    device:      torch.device = torch.device('cpu'),
) -> tuple:
    """
    Load any torchvision model with PAFGraphManager.

    Returns:
        (model, manager)
    """
    import torchvision.models as models

    if not hasattr(models, model_name):
        raise ValueError(f"Unknown model '{model_name}'.")

    weights = 'DEFAULT' if pretrained else None
    model   = getattr(models, model_name)(weights=weights).to(device).eval()
    manager = PAFGraphManager(model)

    arch = 'ViT' if 'vit' in model_name.lower() else 'CNN'
    print(f"✓ Loaded {model_name} ({arch}, pretrained={pretrained})")

    return model, manager
