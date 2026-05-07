"""
PAF Hook Manager — Activation capture and graph tracing for CNN and ViT models.
Supports: ResNet, VGG, and torchvision ViT (vit_b_16, vit_b_32, vit_l_16, etc.)
"""

import torch
import torch.fx
import torch.nn as nn
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
import torchvision.models as models
import operator

# ============================================================================
# Node type classification — single source of truth for CNN + ViT
# ============================================================================
def classify_node(
    node:       torch.fx.Node,
    module_map: Dict[str, nn.Module],
    activations: Dict[str, torch.Tensor] = None,  # optional — for tensor check
) -> str:
    """
    Map FX node to PAF semantic type.
    Any node with multiple tensor predecessors is classified as 'add'
    regardless of its operation name — it joins multiple attribution paths.
    """

    # First determine operation type
    if node.op == 'placeholder':
        return 'input'
    if node.op == 'output':
        return 'output'
    if node.op == 'get_attr':
        return 'parameter'   # learned parameter — leaf node, no predecessors

    base_type = _classify_op(node, module_map)

    # All return values of _classify_op that are NOT generic
    # Generic types that CAN be overridden by predecessor count:
    GENERIC_TYPES = {
        'passthrough','unknown', 'reshape', 'permute',
        'relu', 'gelu', 'dropout', 'softmax', 'add','arithop'
    }

    # Anything not in GENERIC_TYPES is specific — trust _classify_op
    if base_type not in GENERIC_TYPES:
        return base_type

    n_tensor = _count_tensor_predecessors(node, activations)
    n_total  = len(_extract_predecessor_nodes(node))

    if n_tensor > 1:
        return 'add'    # multiple tensor inputs — mass must be split

    if  n_total > 1:
        return 'join'   # one tensor + scalars — passthrough to tensor

    return base_type

def _classify_op(node: torch.fx.Node, module_map: Dict[str, nn.Module]) -> str:
    """Pure operation classification — ignoring graph topology."""
    if node.op == 'call_module':
        module = module_map.get(node.name)
        if module is None:
            return 'passthrough'
        if isinstance(module, nn.MultiheadAttention):   return 'mhsa'
        if isinstance(module, nn.LayerNorm):            return 'layernorm'
        if isinstance(module, nn.GELU):                 return 'gelu'
        if isinstance(module, nn.SiLU):                 return 'gelu'
        if isinstance(module, nn.Conv2d):               return 'conv'
        if isinstance(module, nn.Conv1d):               return 'conv'
        if isinstance(module, nn.Conv3d):               return 'conv'
        if isinstance(module, nn.Linear):               return 'linear'
        if isinstance(module, nn.MaxPool1d):            return 'maxpool'
        if isinstance(module, nn.MaxPool2d):            return 'maxpool'
        if isinstance(module, nn.MaxPool3d):            return 'maxpool'
        if isinstance(module, nn.AvgPool1d):            return 'avgpool'
        if isinstance(module, nn.AvgPool2d):            return 'avgpool'
        if isinstance(module, nn.AvgPool3d):            return 'avgpool'
        if isinstance(module, nn.AdaptiveAvgPool1d):    return 'avgpool'
        if isinstance(module, nn.AdaptiveAvgPool2d):    return 'avgpool'
        if isinstance(module, nn.AdaptiveAvgPool3d):    return 'avgpool'
        if isinstance(module, nn.BatchNorm1d):          return 'batchnorm'
        if isinstance(module, nn.BatchNorm2d):          return 'batchnorm'
        if isinstance(module, nn.BatchNorm3d):          return 'batchnorm'
        if isinstance(module, nn.ReLU):                 return 'relu'
        if isinstance(module, nn.LeakyReLU):            return 'relu'
        if isinstance(module, nn.Dropout):              return 'dropout'
        if isinstance(module, nn.Dropout2d):            return 'dropout'
        if isinstance(module, nn.Identity):             return 'passthrough'
        
        #return type(module).__name__.lower()
        return 'passthrough'

    # Function / method-based classification
    if node.op in ('call_function', 'call_method'):
        target     = node.target
        target_str = str(target)
        
        # --- Arithmetic ---
        if target in (torch.add, operator.add, 'add'):     return 'arithop'
        if target in (torch.mul, operator.mul, 'mul',
                      torch.sub, operator.sub, 'sub'):     return 'arithop'
        
        # --- Activation functions ---
        if 'relu' in target_str:                           return 'relu'
        if 'gelu' in target_str:                           return 'gelu'
        if 'silu' in target_str:                           return 'gelu'
        if 'sigmoid' in target_str:                        return 'passthrough'
        if 'tanh' in target_str:                           return 'passthrough'    
       
        # --- Pooling ---
        if 'max_pool' in target_str:                       return 'maxpool'
        if 'avg_pool' in target_str:                       return 'avgpool'
        if 'adaptive_avg' in target_str:                   return 'avgpool'
        
        # --- Shape operations ---
        if target in ('flatten', torch.flatten):           return 'flatten'
        if target in ('reshape', 'view', 'contiguous'):    return 'reshape'
        if target in ('permute', 'transpose'):             return 'permute'
        if target == 'expand':                             return 'passthrough'
        if target in ('squeeze', 'unsqueeze'):             return 'reshape'
        if target in ('chunk', 'split'):                   return 'reshape'
        if target == torch.cat:                            return 'cat'
        if target in (torch.stack,):                       return 'cat'

        # --- Normalisation ---
        if target in ('softmax', torch.softmax,
                      torch.nn.functional.softmax):         return 'softmax'
        if target in ('layer_norm',
                      torch.nn.functional.layer_norm):       return 'layernorm'
        
        # --- Matrix operations ---
        if target in (torch.matmul, torch.bmm,
                      operator.matmul, '@'):                   return 'passthrough'
        # --- Passthrough operations ---
        if target in ('contiguous', 'clone',
                      'detach', 'to', 'type',
                      'float', 'half', 'double'):       return 'passthrough'
        # Global reductions — passthrough (no layer object available)
        if target in (torch.mean, torch.sum,
                      torch.max, torch.min,
                      'mean', 'sum', 'max', 'min'):     return 'passthrough'
        
        # Scalar arithmetic — produce integers/floats not tensors
        # Transparent to PAF — treated as passthrough
        SCALAR_OPS = {
            operator.floordiv, operator.truediv, operator.mod,
            operator.pow, operator.neg, operator.abs,
            operator.lt, operator.le, operator.gt, operator.ge,
            operator.eq, operator.ne,
            'floordiv', 'truediv', 'mod', 'pow',
        }
        if target in SCALAR_OPS:
            return 'passthrough'
        
        if target in ('getitem', operator.getitem):
            args = node.args
            if len(args) >= 2:
                src = args[0]
                idx = args[1]
                if isinstance(src, torch.fx.Node):
                    src_module = module_map.get(src.name)
                    if isinstance(src_module, nn.MultiheadAttention):
                        return 'mhsa_output' if idx == 0 else 'mhsa_attn_weights'
                if isinstance(idx, tuple) and len(idx) >= 2:
                    if idx[1] == 0:
                        return 'cls_token'
            return 'reshape'
        return 'passthrough'


def _count_tensor_predecessors(
    node:        torch.fx.Node,
    activations: Optional[Dict[str, torch.Tensor]],
) -> int:
    """
    Count predecessors that produce tensor activations.

    With activations (after forward pass):
        Only counts predecessors whose activation is a torch.Tensor.
        Scalars (x.shape[0]), None outputs are excluded.

    Without activations (at trace time):
        Conservative estimate — excludes known scalar-producing ops.
        get_attr nodes counted since they produce parameter tensors.

    Must be consistent with _prev_activation which uses the same
    tensor-vs-scalar distinction for routing decisions.
    """
    count = 0

    for pred in _extract_predecessor_nodes(node):
        if pred.op == 'output':
            continue

        if activations is not None:
            # Ground truth — check actual activation type
            act = activations.get(pred.name)
            if isinstance(act, torch.Tensor):
                count += 1
            # None or scalar (int, float) → skip

        else:
            # Trace time — no activations available
            # Use op type as proxy

            if pred.op == 'get_attr':
                # Parameter tensors — always count
                count += 1

            elif pred.op == 'placeholder':
                # Input tensor — always count
                count += 1

            elif pred.op in ('call_module', 'call_function', 'call_method'):
                # Exclude known scalar-producing ops
                target_str = str(pred.target)
                scalar_ops = {
                    'size', 'shape', '__len__', 'item',
                    'numel', 'dim', 'getitem',
                }
                # getitem on shape tuples produces scalars
                # but getitem on tensors produces tensors —
                # conservatively exclude getitem at trace time
                # it will be corrected after forward pass
                is_scalar = any(s in target_str for s in scalar_ops)
                if not is_scalar:
                    count += 1

    return count
'''
def classify_node(node: torch.fx.Node, module_map: Dict[str, nn.Module]) -> str:
    """
    Map an FX node to a PAF semantic type.
    Covers both CNN (ResNet, VGG) and ViT layer types.
    """
    if node.op == 'placeholder':
        return 'input'

    if node.op == 'output':
        return 'output'

    # Module-based classification — most reliable
    if node.op == 'call_module':
        module = module_map.get(node.name)
        if module is None:
            return 'unknown'

        if isinstance(module, nn.MultiheadAttention):   return 'mhsa'
        if isinstance(module, nn.LayerNorm):            return 'layernorm'
        if isinstance(module, nn.GELU):                 return 'gelu'
        if isinstance(module, nn.SiLU):                 return 'gelu'      # treat same
        if isinstance(module, nn.Conv2d):               return 'conv'
        if isinstance(module, nn.Conv1d):               return 'conv'
        if isinstance(module, nn.Conv3d):               return 'conv'
        if isinstance(module, nn.Linear):               return 'linear'
        if isinstance(module, nn.MaxPool1d):            return 'maxpool'
        if isinstance(module, nn.MaxPool2d):            return 'maxpool'
        if isinstance(module, nn.MaxPool3d):            return 'maxpool'
        if isinstance(module, nn.AvgPool1d):            return 'avgpool'
        if isinstance(module, nn.AvgPool2d):            return 'avgpool'
        if isinstance(module, nn.AvgPool3d):            return 'avgpool'
        if isinstance(module, nn.AdaptiveAvgPool1d):    return 'avgpool'
        if isinstance(module, nn.AdaptiveAvgPool2d):    return 'avgpool'
        if isinstance(module, nn.AdaptiveAvgPool3d):    return 'avgpool'
        if isinstance(module, nn.BatchNorm1d):          return 'batchnorm'
        if isinstance(module, nn.BatchNorm2d):          return 'batchnorm'
        if isinstance(module, nn.BatchNorm3d):          return 'batchnorm'
        if isinstance(module, nn.ReLU):                 return 'relu'
        if isinstance(module, nn.LeakyReLU):            return 'relu'
        if isinstance(module, nn.Dropout):              return 'dropout'
        if isinstance(module, nn.Dropout2d):            return 'dropout'
        if isinstance(module, nn.Identity):             return 'passthrough'

        # Fallback — use class name
        return type(module).__name__.lower()

    # Function / method-based classification
    if node.op in ('call_function', 'call_method'):
        target = node.target
        target_str = str(target)

        # Arithmetic
        if target in (torch.add, operator.add, 'add'):          return 'add'
        if target in (torch.mul, operator.mul, 'mul'):          return 'mul'

        # Activation functions
        if 'relu' in target_str:                                return 'relu'
        if 'gelu' in target_str:                               return 'gelu'
        if 'silu' in target_str:                               return 'gelu'

        # Pooling
        if 'max_pool' in target_str:                           return 'maxpool'
        if 'avg_pool' in target_str or 'adaptive_avg' in target_str:
            return 'avgpool'

        # Shape operations — ViT specific
        if target in ('flatten', torch.flatten):               return 'flatten'
        if target in ('reshape', 'view', 'contiguous'):        return 'reshape'
        if target == 'permute':                                return 'permute'
        if target == 'transpose':                              return 'permute'
        if target == torch.cat:                                return 'cat'
        if target == 'expand':                                 return 'passthrough'
        if target == 'softmax' or target == torch.softmax:     return 'softmax'

        # CLS token extraction: x[:, 0] or x[:, 0, :]
        if target == 'getitem' or target == operator.getitem:
            args = node.args
            if len(args) >= 2:
                idx = args[1]
                # Detect x[:, 0] pattern
                if isinstance(idx, tuple) and len(idx) >= 2:
                    if idx[1] == 0 or idx[1] == slice(None, 1, None):
                        return 'cls_token'
            return 'reshape'

        if hasattr(target, '__name__'):
            return target.__name__.lower()

    return 'unknown'
'''

# ============================================================================
# FX Interpreter — captures all node activations in one forward pass
# ============================================================================

class PAFTracer(torch.fx.Interpreter):
    """
    Runs the FX graph and captures every node's output tensor.
    Much more reliable than register_forward_hook for functional ops
    like add, reshape, cat — which hooks cannot capture.
    """

    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(graph_module)
        self.activations: Dict[str, torch.Tensor] = {}

    def run_node(self, node: torch.fx.Node):
        result = super().run_node(node)
        if isinstance(result, torch.Tensor):
            self.activations[node.name] = result.detach()
        elif isinstance(result, (tuple, list)):
            # MultiheadAttention returns (output, attn_weights)
            # Store just the output tensor at index 0
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    key = node.name if i == 0 else f"{node.name}_{i}"
                    self.activations[key] = r.detach()
        return result

    def clear(self):
        self.activations.clear()


# ============================================================================
# Graph tracing — builds PAF graph info once at construction
# ============================================================================

def _extract_predecessor_nodes(node: torch.fx.Node) -> List[torch.fx.Node]:
    """Recursively extract all FX Node inputs from args and kwargs."""
    nodes = []
    def _recurse(obj):
        if isinstance(obj, torch.fx.Node):
            nodes.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                _recurse(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                _recurse(x)
    _recurse(node.args)
    _recurse(node.kwargs)
    return nodes

def trace_model(
    model:       nn.Module,
    activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    traced = torch.fx.symbolic_trace(model)

    module_map = {}
    named_modules = dict(model.named_modules())
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            try:
                module_map[node.name] = model.get_submodule(node.target)
            except AttributeError:
                module_map[node.name] = named_modules.get(node.target)

    predecessors = defaultdict(list)
    successors   = defaultdict(list)
    node_types   = {}

    for node in traced.graph.nodes:
        if node.op == 'output':
            continue

        # Pass activations so multi-predecessor detection works correctly
        node_types[node.name] = classify_node(node, module_map, activations)

        for inp in _extract_predecessor_nodes(node):
            if inp.op != 'output':
                predecessors[node.name].append(inp.name)
                successors[inp.name].append(node.name)

    all_nodes      = [n.name for n in traced.graph.nodes if n.op != 'output']
    forward_order  = all_nodes
    output_node = list(reversed(forward_order))[0]
    backward_order = build_backward_order_simple_reverse(
        forward_order = forward_order,
        predecessors  = dict(predecessors),
        output_node   = output_node,
    )
    node_map = {
        node.name: node
        for node in traced.graph.nodes
        if node.op != 'output'
    }
    
    pure_parameter_nodes = _compute_pure_parameter_nodes(
        all_nodes    = all_nodes,
        predecessors = predecessors,
        node_types   = node_types,
    )

    return {
        'traced'         : traced,
        'backward_order' : backward_order,
        'forward_order'  : forward_order,
        'predecessors'   : dict(predecessors),
        'successors'     : dict(successors),
        'node_types'     : node_types,
        'module_map'     : module_map,
        'node_map'      : node_map,
        'pure_parameter_nodes' : pure_parameter_nodes,
    }

def _compute_pure_parameter_nodes(
    all_nodes:    List[str],
    predecessors: Dict[str, List[str]],
    node_types:   Dict[str, str],
) -> Set[str]:
    """
    Compute nodes that carry no tensor data from the input image.
    Uses node types only — no activations needed.

    Scalar node types that break the tensor path:
      parameter — get_attr (class_token, pos_embedding etc.)
      unknown   — unrecognised ops (often scalar shape ops)

    Any node not reachable from input via tensor-typed nodes
    is a pure parameter node.
    """

    # Node types that carry tensor data — following these edges is valid
    TENSOR_TYPES = {
        'input', 'conv', 'linear', 'mhsa', 'mhsa_output',
        'relu', 'gelu', 'batchnorm', 'layernorm', 'dropout',
        'avgpool', 'maxpool', 'add', 'cat', 'join',
        'flatten', 'reshape', 'permute', 'passthrough',
        'softmax', 'cls_token',
    }

    # Node types that break the tensor path — do not follow
    SCALAR_TYPES = {
        'parameter',   # get_attr — learned constant
        'unknown',     # unrecognised — conservative: treat as scalar
    }

    # Find input nodes
    input_nodes = [
        n for n in all_nodes
        if node_types.get(n, 'unknown') == 'input'
    ]

    # Build successors from predecessors
    successors: Dict[str, List[str]] = defaultdict(list)
    for node, preds in predecessors.items():
        for pred in preds:
            successors[pred].append(node)

    # BFS from input — follow tensor-typed nodes only
    reachable_via_tensor: Set[str] = set()
    stack = list(input_nodes)

    while stack:
        node = stack.pop()
        if node in reachable_via_tensor:
            continue

        node_type = node_types.get(node, 'unknown')

        # Stop at scalar/parameter nodes — do not follow their successors
        if node_type in SCALAR_TYPES:
            continue

        # Only follow if this is a tensor-typed node
        if node_type not in TENSOR_TYPES:
            continue

        reachable_via_tensor.add(node)

        for succ in successors.get(node, []):
            if succ not in reachable_via_tensor:
                stack.append(succ)

    return set(all_nodes) - reachable_via_tensor

'''
def trace_model(model: nn.Module) -> Dict:
    """
    Symbolically trace the model and extract PAF graph structure.

    Returns dict with:
        traced:         FX GraphModule
        backward_order: list of node names from output → input
        predecessors:   {node_name: [pred_name, ...]}
        successors:     {node_name: [succ_name, ...]}
        node_types:     {node_name: semantic_type_str}
        module_map:     {node_name: nn.Module}
    """
    traced = torch.fx.symbolic_trace(model)

    predecessors: Dict[str, List[str]]  = defaultdict(list)
    successors:   Dict[str, List[str]]  = defaultdict(list)
    node_types:   Dict[str, str]        = {}
    module_map:   Dict[str, nn.Module]  = {}

    # Build module map — node.name → nn.Module
    # Use node.target (module path) to look up via get_submodule
    named_modules = dict(model.named_modules())
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            try:
                module_map[node.name] = model.get_submodule(node.target)
            except AttributeError:
                module_map[node.name] = named_modules.get(node.target)

    # Build graph structure
    for node in traced.graph.nodes:
        if node.op == 'output':
            continue

        node_types[node.name] = classify_node(node, module_map)

        for inp in _extract_predecessor_nodes(node):
            if inp.op != 'output':
                predecessors[node.name].append(inp.name)
                successors[inp.name].append(node.name)

    # Backward order: reverse of forward topological order
    forward_order  = [n.name for n in traced.graph.nodes if n.op != 'output']
    b_order = list(reversed(forward_order))
    all_nodes     = [n.name for n in traced.graph.nodes if n.op != 'output']
    backward_order = build_backward_order_simple_reverse(
        predecessors=dict(predecessors),
        forward_order  = all_nodes,
        output_node=b_order[0]
    )
    
    return {
        'traced':        traced,
        'backward_order': backward_order,
        'predecessors':  dict(predecessors),
        'successors':    dict(successors),
        'node_types':    node_types,
        'module_map':    module_map,
    }
'''

def build_backward_order_simple_reverse(
    forward_order: List[str],
    predecessors: Dict[str, List[str]],
    output_node: str
) -> List[str]:
    """Simple reverse + cleanup unreachable nodes"""
    
    # First, get all reachable nodes from output
    reachable = _get_reachable_nodes(predecessors, output_node)
    
    # Filter forward_order to only reachable nodes, then reverse
    reachable_forward = [n for n in forward_order if n in reachable]
    backward_order = list(reversed(reachable_forward))
    
    return backward_order

def build_backward_order_proper(
    predecessors: Dict[str, List[str]],   # node → predecessors
    all_nodes: List[str],
    output_node: str
) -> List[str]:
    
    # Build successors
    successors = defaultdict(list)
    for node, preds in predecessors.items():
        for pred in preds:
            successors[pred].append(node)

    # Count remaining successors
    remaining = {node: len(successors.get(node, [])) for node in all_nodes}

    visited = set()
    order = []
    queue = deque([output_node])        # Start from output

    while queue:
        node = queue.popleft()

        if node in visited:
            continue

        if remaining[node] > 0:
            continue

        visited.add(node)
        order.append(node)

        # Push predecessors when ready
        for pred in predecessors.get(node, []):
            if pred not in visited:
                remaining[pred] -= 1
                if remaining[pred] == 0:
                    queue.append(pred)

    # Add unreachable nodes at the end
    unreachable = [n for n in all_nodes if n not in visited]
    order.extend(unreachable)

    return order

def _get_reachable_nodes(
    predecessors: Dict[str, List[str]],
    start: str,
) -> Set[str]:
    """
    Return all nodes reachable from start by following predecessors
    (i.e. all ancestors in the forward graph).
    """
    if not isinstance(predecessors, dict):
        raise TypeError(f"predecessors should be a dict, got {type(predecessors)}")
    
    reachable = set()
    stack     = [start]

    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for pred in predecessors.get(node, []):
            if pred not in reachable:
                stack.append(pred)

    return reachable


def _build_backward_order(
    predecessors: Dict[str, List[str]],
    successors:   Dict[str, List[str]],
    all_nodes:    List[str],
    output_node:  str,
) -> List[str]:
    """
    Build backward traversal order (output → input).

    A node is visited only when ALL its forward-successors have been visited.
    Uses a ready-queue — nodes are enqueued exactly once, when they become ready.
    No re-queueing.
    """
    reachable          = _get_reachable_nodes(predecessors, output_node)
    unreachable        = set(all_nodes) - reachable

    # Count pending (unvisited) successors per node
    # Only count successors that are reachable — unreachable ones are ignored
    pending: Dict[str, int] = {
        node: sum(1 for s in successors.get(node, []) if s in reachable)
        for node in reachable
    }

    visited: Set[str] = set(unreachable)   # treat unreachable as already done
    order:   List[str] = []

    # Seed: output node has no successors — immediately ready
    stack = [output_node]

    while stack:
        node = stack.pop()       # LIFO — depth-first, matches PAF traversal intent

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        # Notify predecessors that one of their successors is done
        for pred in predecessors.get(node, []):
            if pred in visited:
                continue
            pending[pred] -= 1
            if pending[pred] == 0:
                # All successors of pred are visited — pred is now ready
                stack.append(pred)

    # Safety net — append any missed nodes
    remaining = [n for n in all_nodes if n not in visited]
    if remaining:
        print(f"WARNING: {len(remaining)} nodes not reached: {remaining[:5]}")
        order.extend(remaining)

    return order

"""
PAF Graph Visualizer — plots the computation graph from predecessors
and the backward order to verify topological correctness.

Two plots:
  1. Forward graph from predecessors (networkx + matplotlib)
  2. Backward order as a linear sequence with edges shown
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import os


# ============================================================================
# Colour map — same as PAFVisualizer branching plot
# ============================================================================

NODE_COLORS = {
    'conv':         '#4a90d9',
    'linear':       '#9b59b6',
    'add':          '#e67e22',
    'relu':         '#2ecc71',
    'batchnorm':    '#e74c3c',
    'layernorm':    '#c0392b',
    'gelu':         '#27ae60',
    'dropout':      '#95a5a6',
    'mhsa':         '#f39c12',
    'maxpool':      '#1abc9c',
    'avgpool':      '#16a085',
    'flatten':      '#8e44ad',
    'reshape':      '#7f8c8d',
    'permute':      '#7f8c8d',
    'cat':          '#d35400',
    'cls_token':    '#e91e63',
    'input':        '#2c3e50',
    'softmax':      '#3498db',
    'unknown':      '#bdc3c7',
}

def _node_color(node_type: str) -> str:
    for key, color in NODE_COLORS.items():
        if key in node_type.lower():
            return color
    return NODE_COLORS['unknown']


# ============================================================================
# Plot 1 — Forward graph from predecessors (hierarchical layout)
# ============================================================================

def plot_forward_graph(
    predecessors: Dict[str, List[str]],
    node_types:   Dict[str, str],
    save_path:    Optional[str] = None,
    title:        str = "Forward Computation Graph (predecessors)",
    max_nodes:    int = 80,
) -> None:
    """
    Plot the forward computation graph derived from predecessors dict.
    Uses a left-to-right hierarchical layout computed via longest-path
    layering (no networkx dependency).

    Each node is coloured by its type. Edges go from predecessor → node.
    """
    # Collect all nodes
    all_nodes = set(predecessors.keys())
    for preds in predecessors.values():
        all_nodes.update(preds)
    all_nodes = list(all_nodes)

    if len(all_nodes) > max_nodes:
        print(f"Graph has {len(all_nodes)} nodes — truncating to first {max_nodes}.")
        all_nodes = all_nodes[:max_nodes]
        predecessors = {
            k: [p for p in v if p in all_nodes]
            for k, v in predecessors.items()
            if k in all_nodes
        }

    # Build successors from predecessors
    successors = defaultdict(list)
    for node, preds in predecessors.items():
        for pred in preds:
            successors[pred].append(node)

    # Assign layers via longest-path from inputs
    layer = {}
    # Input nodes: no predecessors
    inputs = [n for n in all_nodes if not predecessors.get(n)]
    queue  = list(inputs)
    for n in inputs:
        layer[n] = 0

    visited = set(inputs)
    while queue:
        node = queue.pop(0)
        for succ in successors.get(node, []):
            new_layer = layer[node] + 1
            if succ not in layer or layer[succ] < new_layer:
                layer[succ] = new_layer
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    # Assign any remaining nodes
    for n in all_nodes:
        if n not in layer:
            layer[n] = 0

    # Group nodes by layer
    layer_groups = defaultdict(list)
    for n in all_nodes:
        layer_groups[layer[n]].append(n)

    n_layers = max(layer_groups.keys()) + 1
    max_per_layer = max(len(v) for v in layer_groups.values())

    # Compute positions
    pos = {}
    for l, nodes in layer_groups.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = l / max(n_layers - 1, 1)
            y = (i + 1) / (n + 1)
            pos[node] = (x, y)

    # Draw
    fig_w = max(16, n_layers * 1.5)
    fig_h = max(8,  max_per_layer * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    # Draw edges
    for node, preds in predecessors.items():
        if node not in pos:
            continue
        for pred in preds:
            if pred not in pos:
                continue
            x0, y0 = pos[pred]
            x1, y1 = pos[node]
            ax.annotate(
                '',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color='#555', lw=0.8, alpha=0.6,
                    connectionstyle='arc3,rad=0.05'
                )
            )

    # Draw nodes
    for node in all_nodes:
        if node not in pos:
            continue
        x, y   = pos[node]
        ntype  = node_types.get(node, 'unknown')
        color  = _node_color(ntype)
        label  = node if len(node) <= 20 else node[:18] + '..'

        circle = plt.Circle((x, y), 0.012, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x, y - 0.025, label,
            ha='center', va='top',
            fontsize=5, color='#222',
            zorder=4,
        )

    # Legend
    seen_types = set(node_types.get(n, 'unknown') for n in all_nodes)
    patches = [
        mpatches.Patch(color=_node_color(t), label=t)
        for t in sorted(seen_types)
    ]
    ax.legend(
        handles=patches, loc='upper left',
        fontsize=7, ncol=2,
        bbox_to_anchor=(0, 1),
    )

    plt.tight_layout()
    _save_or_show(fig, save_path, 'forward_graph')


# ============================================================================
# Plot 2 — Backward order as ranked sequence
# ============================================================================

def plot_backward_order(
    backward_order: List[str],
    predecessors:   Dict[str, List[str]],
    node_types:     Dict[str, str],
    save_path:      Optional[str] = None,
    title:          str = "Backward Traversal Order",
    max_nodes:      int = 80,
) -> None:
    """
    Plot backward order as a horizontal ranked sequence.
    Each node shown in order left (output) to right (input).
    Edges from predecessor graph are overlaid to verify correctness:
      GREEN edge: predecessor appears LATER in backward order (correct —
                  predecessor is processed after current node)
      RED edge:   predecessor appears EARLIER (wrong — means distribution
                  would not be set when needed)
    """
    order = backward_order[:max_nodes]
    rank  = {node: i for i, node in enumerate(order)}
    n     = len(order)

    # Layout: nodes evenly spaced horizontally, staggered vertically
    xs = np.linspace(0, 1, n)
    ys = np.array([0.5 + 0.15 * ((-1) ** i) * (0.3 if i % 4 < 2 else 0.1)
                   for i in range(n)])

    fig_w = max(20, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    # Draw edges — check ordering correctness
    violations = 0
    for i, node in enumerate(order):
        for pred in predecessors.get(node, []):
            if pred not in rank:
                continue
            j = rank[pred]
            # Correct: pred has higher rank (appears later = processed after)
            correct = j > i
            color   = '#2ecc71' if correct else '#e74c3c'
            lw      = 0.6
            if not correct:
                violations += 1
                lw = 1.5

            ax.annotate(
                '',
                xy     = (xs[j], ys[j]),
                xytext = (xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    alpha=0.5,
                    connectionstyle='arc3,rad=0.3',
                )
            )

    # Draw nodes
    for i, node in enumerate(order):
        ntype = node_types.get(node, 'unknown')
        color = _node_color(ntype)
        label = node if len(node) <= 18 else node[:16] + '..'

        circle = plt.Circle((xs[i], ys[i]), 0.008, color=color, zorder=3)
        ax.add_patch(circle)

        # Rank number above node
        ax.text(
            xs[i], ys[i] + 0.05,
            str(i),
            ha='center', va='bottom',
            fontsize=5, color='#666',
        )
        # Node name below
        ax.text(
            xs[i], ys[i] - 0.05,
            label,
            ha='center', va='top',
            fontsize=4.5, color='#222',
            rotation=45,
        )

    # Summary
    status = "✓ CORRECT" if violations == 0 else f"✗ {violations} VIOLATIONS"
    color  = '#2ecc71' if violations == 0 else '#e74c3c'
    ax.text(
        0.5, 0.97, status,
        ha='center', va='top',
        fontsize=12, fontweight='bold', color=color,
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.03,
        "← output (index 0)          input (index N) →",
        ha='center', va='bottom', fontsize=8, color='#555',
        transform=ax.transAxes,
    )

    # Legend
    patches = [
        mpatches.Patch(color='#2ecc71', label='correct edge (pred processed after)'),
        mpatches.Patch(color='#e74c3c', label='violation (pred processed before)'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path, 'backward_order')

    if violations > 0:
        print(f"\nBackward order has {violations} violation(s).")
        print("Violations mean a predecessor is processed before its successor —")
        print("PAF will fail because the distribution is not yet set.")
    else:
        print(f"\nBackward order is topologically correct ({n} nodes checked).")


# ============================================================================
# Combined — both plots in one call
# ============================================================================

def verify_graph(
    graph_info: Dict,
    save_dir:   str = "PAF-output",
    prefix:     str = "",
    max_nodes:  int = 80,
) -> None:
    """
    Plot both forward graph and backward order for verification.
    Call after hook_manager.run_forward(x) to verify graph structure.

    Args:
        graph_info: model.graph_info dict from PAFHookManager
        save_dir:   directory to save plots
        prefix:     filename prefix (e.g. 'vit_b16' or 'resnet18')
        max_nodes:  truncate large graphs for readability
    """
    os.makedirs(save_dir, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    predecessors   = graph_info['predecessors']
    node_types     = graph_info['node_types']
    backward_order = graph_info['backward_order']

    print(f"Graph: {len(backward_order)} nodes")
    print(f"Plotting forward graph → {save_dir}/{p}forward_graph.png")
    plot_forward_graph(
        predecessors = predecessors,
        node_types   = node_types,
        save_path    = os.path.join(save_dir, f"{p}forward_graph.png"),
        title        = f"{prefix} Forward Computation Graph",
        max_nodes    = max_nodes,
    )

    print(f"Plotting backward order → {save_dir}/{p}backward_order.png")
    plot_backward_order(
        backward_order = backward_order,
        predecessors   = predecessors,
        node_types     = node_types,
        save_path      = os.path.join(save_dir, f"{p}backward_order.png"),
        title          = f"{prefix} Backward Traversal Order",
        max_nodes      = max_nodes,
    )


# ============================================================================
# Helper
# ============================================================================

def _save_or_show(fig, save_path: Optional[str], default_name: str) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.savefig(f"PAF-output/{default_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

'''
def _build_backward_order(
    predecessors: Dict[str, List[str]],   
    successors: Dict[str, List[str]],   
    all_nodes: List[str],
    output_node: str,
) -> List[str]:
    """
    Build backward traversal order (output → input) using stack-based post-order traversal.
    """
    reachable = _get_reachable_nodes(predecessors, output_node)   # We'll define this below

    # Nodes not reachable from output are considered "already visited"
    virtually_visited = set(all_nodes) - reachable

    visited: Set[str] = set(virtually_visited)
    order: List[str] = []
    queue = deque([output_node])

    while queue:
        node = queue.popleft()
        print(f"Node: {node}")
        if node in visited:
            continue

        # Check if ALL reachable forward-successors have been visited
        all_succ_visited = all(
            succ in visited for succ in successors.get(node, [])
        )

        if not all_succ_visited:
            # Not ready — push back onto stack
            queue.append(node)
            continue

        # All relevant successors are done → visit this node
        visited.add(node)
        order.append(node)

        # Push predecessors
        for pred in predecessors.get(node, []):
            if pred not in visited:
                print(f"In Queue: {pred}")
                queue.append(pred)

    # Append any remaining nodes just in case
    remaining = [n for n in all_nodes if n not in visited]
    if remaining:
        order.extend(remaining)

    return order

def _get_reachable_nodes(predecessors: Dict[str, List[str]], start: str) -> Set[str]:
    """Return all nodes reachable from 'start' by following successors (forward graph)."""
    reachable = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        for succ in predecessors.get(node, []):
            if succ not in reachable:
                queue.append(succ)

    return reachable
'''

# ============================================================================
# PAFHookManager — main interface
# ============================================================================

class PAFHookManager:
    """
    Manages activation capture and graph info for PAF.

    Supports CNN (ResNet, VGG) and ViT (torchvision vit_b_16 etc.)
    Uses FX Interpreter for reliable capture of ALL node types including
    functional ops (add, cat, reshape, cls_token slice).

    Usage:
        model, hook_manager = load_model_with_hooks('vit_b_16')
        output = hook_manager.run_forward(x)
        activations = hook_manager.activations   # {node_name: tensor}
        graph_info  = model.graph_info
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        # Trace once at construction — reused for all forward passes
        graph_info = trace_model(model)
        self.model.graph_info = graph_info
        self._tracer = PAFTracer(graph_info['traced'])

        # Expose activations directly
        self.activations: Dict[str, torch.Tensor] = {}
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _refresh(self,model):
        self.model = model
        self.model.eval()
        self._tracer.clear()
        self.activations={}

    def run_forward(self, model, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass and capture all node activations.
        Uses FX Interpreter — no register_forward_hook needed.
        """
        self._refresh(model)
        with torch.no_grad():
            output = self._tracer.run(x)

        self.activations = self._tracer.activations
        self.model.activations = self.activations
        self._update_node_types_with_activations()
        return output

    def _update_node_types_with_activations(self) -> None:
        """
        Update node_types after forward pass when activations are known.
        Reclassifies nodes with multiple tensor predecessors as 'add'.
        """
        graph_info   = self.model.graph_info
        module_map   = graph_info['module_map']
        activations  = self.activations

        updated = []
        for node in graph_info['traced'].graph.nodes:
            if node.op == 'output':
                continue
            old_type = graph_info['node_types'].get(node.name, 'unknown')
            new_type = classify_node(node, module_map, activations)
            if old_type != new_type:
                updated.append(f"{node.name}: {old_type} → {new_type}")
            graph_info['node_types'][node.name] = new_type
        
        graph_info['pure_parameter_nodes'] = _compute_pure_parameter_nodes(
            all_nodes    = graph_info['forward_order'],
            predecessors = graph_info['predecessors'],
            node_types   = graph_info['node_types'],
        )
        '''
        if updated:
            print(f"Node types updated after forward pass ({len(updated)}):")
            for u in updated[:10]:
                print(f"  {u}")
            if len(updated) > 10:
                print(f"  ... and {len(updated) - 10} more") 
        '''   
    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def graph_info(self) -> Dict:
        return self.model.graph_info

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def clear_activations(self) -> None:
        self.activations.clear()
        self._tracer.clear()
        if hasattr(self.model, 'activations'):
            self.model.activations.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(self, x: torch.Tensor) -> None:
        """
        Run forward pass and print capture status for every node.
        Use this to verify all layers are captured correctly.
        """
        self.run_forward(x)
        graph_info = self.model.graph_info

        print(f"\n{'Status':<6} {'Node Name':<45} {'Type':<15} {'Shape'}")
        print("-" * 90)

        missing = 0
        for node in graph_info['traced'].graph.nodes:
            if node.op == 'output':
                continue
            name      = node.name
            ntype     = graph_info['node_types'].get(name, 'unknown')
            act       = self.activations.get(name)
            shape_str = str(tuple(act.shape)) if act is not None else 'MISSING'
            status    = "✓" if act is not None else "✗"
            if act is None:
                missing += 1
            print(f"{status:<6} {name:<45} {ntype:<15} {shape_str}")

        print(f"\nTotal: {len(self.activations)} captured, {missing} missing")

    def print_graph(self) -> None:
        """Print the computation graph with node types."""
        graph_info = self.model.graph_info
        print(f"\n{'Node Name':<45} {'Type':<15} {'Predecessors'}")
        print("-" * 90)
        for name in reversed(graph_info['backward_order']):
            ntype = graph_info['node_types'].get(name, '?')
            preds = graph_info['predecessors'].get(name, [])
            print(f"{name:<45} {ntype:<15} {preds}")


# ============================================================================
# Convenience loaders
# ============================================================================

def load_model_with_hooks(
    model_name: str = 'resnet18',
    pretrained:  bool = True,
    device:      torch.device = torch.device('cpu'),
) -> tuple:
    """
    Load any torchvision model with PAF hooks.
    Supports: resnet18/34/50/101, vgg11/13/16/19,
              vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

    Returns:
        model:        nn.Module (eval mode, graph_info attached)
        hook_manager: PAFHookManager
    """
    if not hasattr(models, model_name):
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: resnet18, resnet50, vit_b_16, vgg16, ..."
        )

    weights_arg = 'DEFAULT' if pretrained else None
    model = getattr(models, model_name)(weights=weights_arg)
    model = model.to(device).eval()

    hook_manager = PAFHookManager(model)

    arch = 'ViT' if 'vit' in model_name.lower() else 'CNN'
    print(f"✓ Loaded {model_name} ({arch}, pretrained={pretrained})")

    return model, hook_manager


# ============================================================================
# DeepSHAP compatibility — unchanged from original
# ============================================================================

def make_model_universal_for_shap(model: nn.Module) -> nn.Module:
    """
    Creates a deepcopy of the model safe for DeepSHAP.
    Replaces shared ReLU instances with unique non-inplace ones.
    Works for ResNet BasicBlock and Bottleneck.
    """
    import copy, types
    from torchvision.models.resnet import BasicBlock, Bottleneck

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
        out = self.bn3(self.conv3(out))
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


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    device = torch.device('cpu')

    # --- ResNet18 ---
    print("\n" + "=" * 60)
    print("ResNet18")
    print("=" * 60)
    model_r, hm_r = load_model_with_hooks('resnet18', pretrained=False, device=device)
    x = torch.randn(1, 3, 224, 224)
    hm_r.run_forward(x)
    hm_r.diagnose(x)

    # --- ViT-B/16 ---
    print("\n" + "=" * 60)
    print("ViT-B/16")
    print("=" * 60)
    model_v, hm_v = load_model_with_hooks('vit_b_16', pretrained=False, device=device)
    hm_v.run_forward(x)
    hm_v.diagnose(x)