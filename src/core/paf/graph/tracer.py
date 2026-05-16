"""
PAF Graph Tracer
================
Traces model architecture ONCE via torch.fx.symbolic_trace.
Result is immutable — does not change when weights or activations change.
Only needs re-running if model architecture changes.
"""

import torch
import torch.fx
import torch.nn as nn
import operator
from collections import defaultdict
from typing import Dict, List, Optional, Set


class GraphInfo:
    """
    Immutable graph structure extracted from FX trace.
    Created once per model architecture — never modified.

    Attributes:
        traced:               FX GraphModule
        forward_order:        node names input → output
        backward_order:       node names output → input (reachable only)
        predecessors:         {node_name: [pred_name, ...]}
        successors:           {node_name: [succ_name, ...]}
        node_types:           {node_name: semantic_type_str}
        module_map:           {node_name: nn.Module}
        node_map:             {node_name: torch.fx.Node}
        pure_parameter_nodes: set of nodes unreachable from input
    """

    def __init__(
        self,
        traced:               torch.fx.GraphModule,
        forward_order:        List[str],
        backward_order:       List[str],
        predecessors:         Dict[str, List[str]],
        successors:           Dict[str, List[str]],
        node_types:           Dict[str, str],
        module_map:           Dict[str, nn.Module],
        node_map:             Dict[str, torch.fx.Node],
        pure_parameter_nodes: Set[str],
    ):
        self.traced               = traced
        self.forward_order        = forward_order
        self.backward_order       = backward_order
        self.predecessors         = predecessors
        self.successors           = successors
        self.node_types           = node_types
        self.module_map           = module_map
        self.node_map             = node_map
        self.pure_parameter_nodes = pure_parameter_nodes

    def update_node_types(
        self,
        activations: Dict[str, torch.Tensor],
        module_map:  Dict[str, nn.Module],
    ) -> None:
        """
        Refine node_types using ground-truth activation types.
        Called ONCE after the first forward pass — not on every call.

        Only nodes whose type changes are updated.
        pure_parameter_nodes is recomputed after reclassification.
        """
        changed = False
        for node in self.traced.graph.nodes:
            if node.op == 'output':
                continue
            old_type = self.node_types.get(node.name, 'unknown')
            new_type = classify_node(node, module_map, activations)
            if old_type != new_type:
                self.node_types[node.name] = new_type
                changed = True

        if changed:
            self.pure_parameter_nodes = _compute_pure_parameter_nodes(
                all_nodes    = self.forward_order,
                predecessors = self.predecessors,
                node_types   = self.node_types,
            )

    def __repr__(self) -> str:
        return (
            f"GraphInfo("
            f"{len(self.forward_order)} nodes, "
            f"{len(self.pure_parameter_nodes)} param nodes, "
            f"backward_order[0]={self.backward_order[0]!r}"
            f")"
        )


def trace_model(model: nn.Module) -> GraphInfo:
    """
    Symbolically trace model architecture and build GraphInfo.
    Runs ONCE per model — result is architecture-dependent only.

    Args:
        model: nn.Module in eval mode

    Returns:
        GraphInfo — immutable graph structure
    """
    model.eval()
    traced = torch.fx.symbolic_trace(model)

    # Build module_map: node.name → nn.Module
    module_map = {}
    named_modules = dict(model.named_modules())
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            try:
                module_map[node.name] = model.get_submodule(node.target)
            except AttributeError:
                module_map[node.name] = named_modules.get(node.target)

    # Build graph structure
    predecessors: Dict[str, List[str]] = defaultdict(list)
    successors:   Dict[str, List[str]] = defaultdict(list)
    node_types:   Dict[str, str]       = {}
    node_map:     Dict[str, torch.fx.Node] = {}

    for node in traced.graph.nodes:
        if node.op == 'output':
            continue
        node_map[node.name]   = node
        # No activations at trace time — conservative classification
        node_types[node.name] = classify_node(node, module_map, activations=None)

        for inp in _extract_predecessors(node):
            if inp.op != 'output':
                predecessors[node.name].append(inp.name)
                successors[inp.name].append(node.name)

    predecessors = dict(predecessors)
    successors   = dict(successors)

    forward_order = [
        n.name for n in traced.graph.nodes if n.op != 'output'
    ]

    # Find output node — last reachable node in forward order
    output_node = forward_order[-1]

    backward_order = _build_backward_order(
        forward_order = forward_order,
        predecessors  = predecessors,
        output_node   = output_node,
    )

    pure_parameter_nodes = _compute_pure_parameter_nodes(
        all_nodes    = forward_order,
        predecessors = predecessors,
        node_types   = node_types,
    )

    return GraphInfo(
        traced               = traced,
        forward_order        = forward_order,
        backward_order       = backward_order,
        predecessors         = predecessors,
        successors           = successors,
        node_types           = node_types,
        module_map           = module_map,
        node_map             = node_map,
        pure_parameter_nodes = pure_parameter_nodes,
    )


# ------------------------------------------------------------------
# Helper functions for graph analysis
# ------------------------------------------------------------------

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

    for pred in _extract_predecessors(node):
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
    n_total  = len(_extract_predecessors(node))

    if n_tensor > 1:
        return 'add'    # multiple tensor inputs — mass must be split

    if  n_total > 1:
        return 'join'   # one tensor + scalars — passthrough to tensor

    return base_type


def _classify_by_module_only(module: nn.Module) -> str:
    """Conservative classification when activation shapes unknown."""
    module_type = type(module).__name__
    
    # CNN layers
    if module_type in ['Conv2d', 'Conv1d', 'Conv3d']:
        return 'conv'
    elif module_type in ['Linear', 'Matmul']:
        return 'linear'
    elif module_type in ['MaxPool2d', 'MaxPool1d', 'MaxPool3d']:
        return 'maxpool'
    elif module_type in ['AvgPool2d', 'AvgPool1d', 'AvgPool3d', 'AdaptiveAvgPool2d']:
        return 'avgpool'
    elif module_type in ['BatchNorm2d', 'BatchNorm1d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm']:
        return 'passthrough'
    elif module_type in ['ReLU', 'ReLU6', 'LeakyReLU', 'ELU', 'GELU', 'Sigmoid', 'Tanh']:
        return 'passthrough'
    elif module_type in ['Dropout', 'Dropout2d', 'Dropout3d']:
        return 'passthrough'
    elif module_type == 'MultiheadAttention':
        return 'mhsa'
    elif module_type in ['Embedding', 'Parameter']:
        return 'parameter'
    else:
        return 'unknown'


def _classify_by_module_and_shape(module: nn.Module, act_shape: torch.Size) -> str:
    """Precise classification using activation shapes."""
    module_type = type(module).__name__
    
    # CNN layers with shape information
    if module_type in ['Conv2d', 'Conv1d', 'Conv3d']:
        return 'conv'
    elif module_type in ['Linear', 'Matmul']:
        return 'linear'
    elif module_type in ['MaxPool2d', 'MaxPool1d', 'MaxPool3d']:
        return 'maxpool'
    elif module_type in ['AvgPool2d', 'AvgPool1d', 'AvgPool3d']:
        return 'avgpool'
    elif module_type == 'AdaptiveAvgPool2d':
        # Distinguish between global avgpool (1x1 output) and adaptive
        if act_shape[-2:] == torch.Size([1, 1]):
            return 'avgpool'
        else:
            return 'adaptive_avgpool'
    elif module_type in ['BatchNorm2d', 'BatchNorm1d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm']:
        return 'passthrough'
    elif module_type in ['ReLU', 'ReLU6', 'LeakyReLU', 'ELU', 'GELU', 'Sigmoid', 'Tanh']:
        return 'passthrough'
    elif module_type in ['Dropout', 'Dropout2d', 'Dropout3d']:
        return 'passthrough'
    elif module_type == 'MultiheadAttention':
        return 'mhsa'
    elif module_type in ['Embedding', 'Parameter']:
        return 'parameter'
    else:
        return 'unknown'


def _extract_predecessors(node: torch.fx.Node) -> List[torch.fx.Node]:
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

def _build_backward_order(
    forward_order: List[str],
    predecessors:  Dict[str, List[str]],
    output_node:   str,
) -> List[str]:
    """
    Build backward traversal order reachable from output.
    Uses DFS from output node following predecessor links.
    """
    reachable = set()
    stack = [output_node]
    
    # Find all nodes reachable from output
    while stack:
        current = stack.pop()
        if current not in reachable:
            reachable.add(current)
            for pred in predecessors.get(current, []):
                if pred not in reachable:
                    stack.append(pred)
    
    # Return reachable nodes in reverse forward order (backwards)
    reachable_forward=[node for node in forward_order if node in reachable]
    backward_order = list(reversed(reachable_forward))
    
    return backward_order


def _compute_pure_parameter_nodes(
    all_nodes:    List[str],
    predecessors: Dict[str, List[str]],
    node_types:   Dict[str, str],
) -> Set[str]:
    """
    Find parameter nodes that are never reachable from input.
    These are pure parameters (learned priors) not derived from input data.
    """
    # Find input nodes
    input_nodes = {node for node, ntype in node_types.items() if ntype == 'input'}
    
    # Find nodes reachable from any input
    reachable_from_input = set()
    stack = list(input_nodes)
    
    while stack:
        current = stack.pop()
        if current not in reachable_from_input:
            reachable_from_input.add(current)
            # Find successors (reverse of predecessors)
            for node, preds in predecessors.items():
                if current in preds and node not in reachable_from_input:
                    stack.append(node)
    
    # Parameter nodes not reachable from input are pure parameters
    parameter_nodes = {node for node, ntype in node_types.items() if ntype == 'parameter'}
    pure_parameters = parameter_nodes - reachable_from_input
    
    return pure_parameters
