"""
PAF Activation Store
=====================
Captures activations via FX Interpreter on each forward pass.
Completely separate from graph tracing — activations change
every forward pass, graph structure never changes.
"""

import torch
import torch.fx
from typing import Dict


class ActivationStore:
    """
    Captures all node activations via FX Interpreter.

    Does NOT own the graph — receives GraphInfo from PAFGraphManager.
    Does NOT mutate the model — stores activations internally.

    Usage:
        store = ActivationStore(graph_info.traced)
        store.run(x)
        activations = store.activations   # {node_name: tensor}
    """

    def __init__(self, traced: torch.fx.GraphModule):
        self._interpreter = _PAFInterpreter(traced)
        self.activations: Dict[str, torch.Tensor] = {}

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass and capture all activations.
        Safe to call multiple times — previous activations are cleared.
        """
        self._interpreter.clear()
        with torch.no_grad():
            output = self._interpreter.run(x)
        self.activations = dict(self._interpreter.activations)
        return output

    def clear(self) -> None:
        self.activations.clear()
        self._interpreter.clear()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.activations[key]

    def get(self, key: str, default=None):
        return self.activations.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.activations

    def __len__(self) -> int:
        return len(self.activations)


class _PAFInterpreter(torch.fx.Interpreter):
    """Internal FX Interpreter — not exposed outside activation_store.py"""

    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(graph_module)
        self.activations: Dict[str, torch.Tensor] = {}

    def run_node(self, node: torch.fx.Node):
        result = super().run_node(node)
        if isinstance(result, torch.Tensor):
            self.activations[node.name] = result.detach()
        elif isinstance(result, (tuple, list)):
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    key = node.name if i == 0 else f"{node.name}_{i}"
                    self.activations[key] = r.detach()
        return result

    def clear(self) -> None:
        self.activations.clear()
