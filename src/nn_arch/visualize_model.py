"""
Build Computational Graph Directly from Model

Traces the forward pass to build predecessor/successor graph
without relying on hook_manager.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torchvision.models as models

from nn_arch.visualize_architecture_graphviz import draw_resnet_no_overlap


class NodeLevelGraphBuilder:
    """
    Build graph showing EVERY node (module) and its connections.
    
    No container grouping - just pure DAG of individual modules.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.predecessors = {}  # node_name -> [predecessor_node_names]
        self.module_map = {}    # node_name -> module object
        self.module_types = {}  # node_name -> module type
    
    def build_graph(self) -> Dict:
        """Build node-level graph"""
        
        print("\n" + "="*80)
        print("BUILDING NODE-LEVEL COMPUTATIONAL GRAPH")
        print("="*80 + "\n")
        
        # Extract all modules
        self._extract_all_modules()
        print(f"✓ Found {len(self.module_map)} modules\n")
        
        # Build connections from Sequential structure
        self._build_sequential_connections()
        print(f"✓ Built sequential connections\n")
        
        # Infer parent-child relationships
        self._build_parent_child_connections()
        print(f"✓ Built parent-child relationships\n")
        
        # Detect skip connections (Add nodes)
        self._detect_skip_connections()
        print(f"✓ Detected skip connections\n")
        
        # Validate graph
        self._validate_graph()
        
        print(f"✓ Complete graph with {len(self.predecessors)} nodes")
        
        return {
            'predecessors': self.predecessors,
            'module_map': self.module_map,
            'module_types': self.module_types,
        }
    
    def _extract_all_modules(self):
        """Extract all modules (except root)"""
        for name, module in self.model.named_modules():
            if name:  # Skip root
                self.module_map[name] = module
                self.module_types[name] = type(module).__name__
    
    def _build_sequential_connections(self):
        """
        For Sequential containers, children are executed in order.
        Infer: child[i] takes from child[i-1]
        """
        
        for parent_name, parent_module in self.model.named_modules():
            if not isinstance(parent_module, nn.Sequential):
                continue
            
            # Get children of this Sequential
            children = []
            for child_name, child_module in parent_module.named_children():
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                children.append((child_name, full_name))
            
            # Sort by index
            try:
                children = sorted(children, key=lambda x: int(x[0]))
            except ValueError:
                pass
            
            # Build chain: child[0] ← parent, child[1] ← child[0], etc.
            for i, (_, full_name) in enumerate(children):
                if i == 0:
                    # First child takes from parent
                    if parent_name:
                        self.predecessors[full_name] = [parent_name]
                    else:
                        self.predecessors[full_name] = ['input']
                else:
                    # Takes from previous child
                    prev_name = children[i-1][1]
                    self.predecessors[full_name] = [prev_name]
    
    def _build_parent_child_connections(self):
        """
        For non-Sequential modules, infer from hierarchy:
        If A.B.C exists, then B.C likely depends on A.B
        """
        
        for node_name in self.module_map.keys():
            if node_name in self.predecessors:
                continue  # Already set
            
            # Get parent
            parts = node_name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name = parts[0]
                
                # If parent exists and is not Sequential, it's likely the predecessor
                if parent_name in self.module_map:
                    parent_type = self.module_types[parent_name]
                    
                    if parent_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
                        self.predecessors[node_name] = [parent_name]
    
    def _detect_skip_connections(self):
        """
        Detect skip connections by finding Add nodes.
        Add nodes merge two paths - both are predecessors.
        """
        
        # First, identify all potential merge points
        merge_nodes = {}  # node_name -> list of potential input paths
        
        for node_name in self.module_map.keys():
            node_type = self.module_types[node_name]
            
            # Look for patterns indicating merges
            if 'add_' in node_name or 'add' in node_name.lower():
                # This is a merge node
                merge_nodes[node_name] = []
            elif node_type in ['BasicBlock', 'Bottleneck']:
                # BasicBlocks have internal Add - look for the merge
                # Usually it's in the relu after add
                for other_name in self.module_map.keys():
                    if node_name in other_name and 'relu' in other_name:
                        # This could be the output of the Add inside the block
                        merge_nodes[other_name] = []
        
        # For each merge node, find what feeds into it
        for merge_name in merge_nodes.keys():
            # Look for nodes with 'bn2' or similar that could feed into the merge
            # Pattern: main_path_output → Add ← skip_path_output
            
            # Find bn2 or conv2 in same block that could be main path
            base_name = merge_name.rsplit('.', 1)[0]  # e.g., "layer1.0"
            
            main_path_outputs = []
            skip_path_outputs = []
            
            for node_name in self.module_map.keys():
                if base_name not in node_name:
                    continue
                
                if 'bn2' in node_name or ('conv2' in node_name and 'bn2' not in node_name):
                    # This could be main path output
                    main_path_outputs.append(node_name)
                elif 'downsample' in node_name:
                    # This is skip path output
                    skip_path_outputs.append(node_name)
            
            # Add both paths as predecessors
            for output in main_path_outputs:
                if output not in self.predecessors.get(merge_name, []):
                    if merge_name not in self.predecessors:
                        self.predecessors[merge_name] = []
                    if output not in self.predecessors[merge_name]:
                        self.predecessors[merge_name].append(output)
            
            for output in skip_path_outputs:
                if output not in self.predecessors.get(merge_name, []):
                    if merge_name not in self.predecessors:
                        self.predecessors[merge_name] = []
                    if output not in self.predecessors[merge_name]:
                        self.predecessors[merge_name].append(output)
    
    def _validate_graph(self):
        """Validate graph completeness"""
        
        print("Graph Validation:")
        
        # Check for input layers (no predecessors or 'input')
        input_layers = [n for n, p in self.predecessors.items() if not p or p == ['input']]
        print(f"  Input layers: {len(input_layers)}")
        
        # Check for missing predecessors
        all_predecessors = set()
        for preds in self.predecessors.values():
            all_predecessors.update(preds)
        
        missing = all_predecessors - set(self.module_map.keys()) - {'input'}
        if missing:
            print(f"  ⚠️  Missing nodes referenced: {missing}")
        
        # Check for output layers (high fan-out)
        max_dependents = 0
        for node in self.module_map.keys():
            dependents = sum(1 for preds in self.predecessors.values() if node in preds)
            max_dependents = max(max_dependents, dependents)
        
        output_layers = [n for n, preds in self.predecessors.items() if n.endswith(('fc', 'avgpool'))]
        print(f"  Output layers (likely): {output_layers}")
        
        print()


def build_node_level_graph(model: nn.Module) -> Dict:
    """
    Build and return node-level graph from model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
        
    Returns
    -------
    graph_info : Dict
        predecessors, module_map, module_types
        
    Example
    -------
    import torchvision.models as models
    from build_graph_from_model import build_node_level_graph
    
    model = models.resnet18(pretrained=True)
    graph_info = build_node_level_graph(model)
    
    # Now predecessors shows node-level connections!
    # Example:
    # 'layer1.0.conv1': ['maxpool']
    # 'layer1.0.bn1': ['layer1.0.conv1']
    # 'layer1.0.relu': ['layer1.0.bn1', 'add_add']  # Skip connection!
    
    # Visualize
    from draw_resnet_no_overlap import draw_resnet_no_overlap
    draw_resnet_no_overlap(
        predecessors=graph_info['predecessors'],
        save_path='resnet_nodes.svg',
        use_graphviz=True
    )
    """
    
    builder = NodeLevelGraphBuilder(model)
    return builder.build_graph()


def print_node_level_graph(graph_info: Dict, layer_filter: str = None) -> None:
    """
    Print node-level graph in readable format.
    
    Parameters
    ----------
    graph_info : Dict
        Output from build_node_level_graph()
    layer_filter : str, optional
        Only show nodes containing this string (e.g., "layer4")
    """
    
    predecessors = graph_info['predecessors']
    module_types = graph_info['module_types']
    
    print("\n" + "="*80)
    print("NODE-LEVEL GRAPH")
    print("="*80 + "\n")
    
    for node_name in sorted(predecessors.keys()):
        if layer_filter and layer_filter not in node_name:
            continue
        
        preds = predecessors[node_name]
        node_type = module_types.get(node_name, 'unknown')
        
        # Format: node_name [type] ← predecessors
        print(f"{node_name:40s} [{node_type:12s}] ← {preds}")

from graphviz import Digraph

def draw_resnet18_layer4():
    dot = Digraph(comment='ResNet18 - Layer 4', format='png')
    dot.attr(rankdir='TB', size='12,10')

    # Main flow
    dot.node('input', 'Input from Layer 3\n(256 channels)', shape='box')
    
    # Block 0 (with downsample)
    dot.node('block0_conv1', 'Conv3x3\nstride=2\n512 ch', shape='box', fillcolor='lightblue', style='filled')
    dot.node('block0_bn1', 'BN', shape='box', fillcolor='lightgreen', style='filled')
    dot.node('block0_relu1', 'ReLU', shape='box', fillcolor='yellow', style='filled')
    dot.node('block0_conv2', 'Conv3x3\nstride=1\n512 ch', shape='box', fillcolor='lightblue', style='filled')
    dot.node('block0_bn2', 'BN', shape='box', fillcolor='lightgreen', style='filled')
    
    # Downsample path for Block 0
    dot.node('ds0', 'Downsample\n1x1 Conv stride=2\n+ BN', shape='box', fillcolor='plum', style='filled')
    
    dot.node('add0', 'Add', shape='circle')
    dot.node('relu0', 'ReLU', shape='box', fillcolor='yellow', style='filled')

    # Block 1 (identity)
    dot.node('block1_conv1', 'Conv3x3\nstride=1\n512 ch', shape='box', fillcolor='lightblue', style='filled')
    dot.node('block1_bn1', 'BN', shape='box', fillcolor='lightgreen', style='filled')
    dot.node('block1_relu1', 'ReLU', shape='box', fillcolor='yellow', style='filled')
    dot.node('block1_conv2', 'Conv3x3\nstride=1\n512 ch', shape='box', fillcolor='lightblue', style='filled')
    dot.node('block1_bn2', 'BN', shape='box', fillcolor='lightgreen', style='filled')
    
    dot.node('add1', 'Add', shape='circle')
    dot.node('relu1', 'ReLU (Output)', shape='box', fillcolor='yellow', style='filled')

    # Connections for Block 0
    dot.edge('input', 'block0_conv1')
    dot.edge('block0_conv1', 'block0_bn1')
    dot.edge('block0_bn1', 'block0_relu1')
    dot.edge('block0_relu1', 'block0_conv2')
    dot.edge('block0_conv2', 'block0_bn2')
    dot.edge('block0_bn2', 'add0')
    
    dot.edge('input', 'ds0')
    dot.edge('ds0', 'add0')
    dot.edge('add0', 'relu0')

    # Connections for Block 1
    dot.edge('relu0', 'block1_conv1')
    dot.edge('block1_conv1', 'block1_bn1')
    dot.edge('block1_bn1', 'block1_relu1')
    dot.edge('block1_relu1', 'block1_conv2')
    dot.edge('block1_conv2', 'block1_bn2')
    dot.edge('block1_bn2', 'add1')
    dot.edge('relu0', 'add1')          # identity shortcut
    dot.edge('add1', 'relu1')

    dot.render('resnet18_layer4_correct', view=True)   # Saves as resnet18_layer4_correct.png and opens it


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
        
    print("\n" + "="*80)
    print("Example: ResNet-18 Node-Level Graph")
    print("="*80 + "\n")
    
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Build graph
    graph_info = build_node_level_graph(model)
    
    # Print all nodes
    print_node_level_graph(graph_info)
    
    # Print only layer4
    print("\n" + "="*80)
    print("Layer 4 Nodes Only")
    print("="*80 + "\n")
    
    print_node_level_graph(graph_info, layer_filter='layer4')
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80 + "\n")
    
    preds = graph_info['predecessors']
    print(f"Total nodes: {len(preds)}")
    
    # Count by type
    types_count = {}
    for node_type in graph_info['module_types'].values():
        types_count[node_type] = types_count.get(node_type, 0) + 1
    
    print("\nNodes by type:")
    for node_type in sorted(types_count.keys()):
        count = types_count[node_type]
        print(f"  {node_type:20s}: {count:3d}")
    
    # 3. Visualize
    draw_resnet_no_overlap(
        predecessors=graph_info['predecessors'],
        save_path='resnet_complete.svg',
        use_graphviz=True
    )
    draw_resnet18_layer4()
