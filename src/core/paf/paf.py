"""
Probabilistic Activation Flow (PAF) — Main Wrapper
====================================================
Connects engine.py, scoring.py, and propagation.py.
 
This file contains only:
  - PAF.__init__   : wires up engine, scoring, propagation, dispatch table
  - PAF.run        : entry point — forward pass + explain
  - PAF.explain    : backward traversal loop
  - PAF._store_edge, _get_dist, _prev_activation : storage helpers
  - PAF.extract_weights, cleanup : utilities
 
All scoring logic   → scoring.py   (build_scoring, ScoringMode)
All engine logic    → engine.py    (AttributionEngine)
All layer logic     → propagation.py (LayerPropagator)
All graph logic     → core/graph_manager.py (PAFGraphManager)
"""
import builtins
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.paf.graph.manager import PAFGraphManager
from core.paf.engine import AttributionEngine
from core.paf.scoring import ScoringMode, build_scoring
from core.paf.propagation import LayerPropagator

# Check if 'profile' is already in builtins (injected by kernprof)
if 'profile' not in builtins.__dict__:
    def profile(func):
        return func
    builtins.profile = profile


class PAF:
    
    def __init__(
        self,
        model:          nn.Module,
        graph_manager:   PAFGraphManager,
        modes:          Optional[List[Tuple[ScoringMode, dict]]] = None,
        scoring_mode:   ScoringMode = ScoringMode.ABS,
        tau:            float       = 1.0,
        eps:            float = 1e-9,
        debug_level:    int = 0,
        output_mode:    str = None,
        target_class: Optional[int]           = None,
        true_class:   Optional[int]           = None,
        x: Optional[torch.Tensor] = None,
        redistribute_param_mass: bool = False,
    ):
        assert eps > 0, "eps must be positive"

        self.model = model
        self.eps = eps
        self.debug_level = debug_level
        self.scaling_factor = 1.0
        self.graph_manager = graph_manager
        
        if output_mode is None:
            self.output_mode = "target"
        else:
            self.output_mode = output_mode
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)
        self.graph_manager.model.to(self.device)
        
        if modes is None:
            modes = [(scoring_mode, {'tau': tau})]
        self._modes = modes

        # Initialize refactored components
        self._engine = AttributionEngine(eps=eps)
       #self._engine.distribute_attribution = self._engine._distribute_attribution
        self._engine.redistribute_param_mass = redistribute_param_mass
        self.propagator = LayerPropagator(graph_info=graph_manager.graph_info, debug_level=debug_level, eps=eps, model=model)
        self.propagator.set_engine(self._engine)
        self.propagator.redistribute_param_mass = redistribute_param_mass
        
        # Results — one dict per mode
        self.distributions: Dict[tuple, Dict[str, torch.Tensor]] = {}
        self.edge_mass: Dict[tuple, Dict] = {}
        self.activations = {}
        self.redistribute_param_mass = redistribute_param_mass
        
        # Build dispatch table once — maps node_type → bound method
        self._dispatch: Dict[str, callable] = {
            # CNN
            'linear'     : self._handle_linear,
            'conv'        : self._handle_conv,
            'maxpool'     : self._handle_maxpool,
            'avgpool'     : self._handle_avgpool,
            'flatten'     : self._handle_flatten,
            # Passthrough
            'relu'        : self._handle_passthrough,
            'batchnorm'   : self._handle_passthrough,
            'layernorm'   : self._handle_passthrough,
            'gelu'        : self._handle_passthrough,
            'dropout'     : self._handle_passthrough,
            'softmax'     : self._handle_passthrough,
            'passthrough' : self._handle_passthrough,
            # ViT shape ops
            'reshape'     : self._handle_reshape,
            'permute'     : self._handle_reshape,
            'parameter'   : self._handle_parameter, 
            'cat'         : self._handle_cat_token,
            'join'        : self._handle_join,
            'cls_token'   : self._handle_cls_token,
            'mhsa'        : self._handle_mhsa,
            'mhsa_output'     : self._handle_passthrough,
            'mhsa_attn_weights': self._handle_passthrough,
            # ADD handled separately
            'add'         : self._handle_add,
            'passthrough'         : self._handle_passthrough,
        }

        # Build scoring functions
        self._score_fns: Dict[tuple, callable] = {}
        for mode, kwargs in modes:
            t = kwargs.get('tau', 1.0)
            alpha = kwargs.get('alpha', 1.0)
            beta = kwargs.get('beta', 0.0)
            if mode == ScoringMode.SIGNED_SPLIT:
                key = (mode, t, alpha, beta)   # unique key per alpha/beta combination
            else:
                key = (mode, t)                # unchanged for all other modes
            kwargs_rest = {k: v for k, v in kwargs.items() if k != 'tau'}
            self._score_fns[key] = build_scoring(mode, t, eps=eps, **kwargs_rest)

            # Initialise result containers per mode
            self.distributions[key] = {}
            self.edge_mass[key] = {}

        if x is not None and target_class is not None and true_class is not None:
            self.run(
                x            = x,
                target_label = target_class,
                true_label   = true_class,
                output_mode  = self.output_mode
            )

    def init_output_distribution(
        self,
        output_logits: torch.Tensor,
        target_class: Optional[int] = None,
        true_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Initialise the probability distribution at the output layer.
        Delegate to propagator which contains the implementation.
        """
        # Set output mode on propagator
        self.propagator.output_mode = self.output_mode
        # Set scaling_factor on propagator for contrastive modes
        self.propagator.scaling_factor = self.scaling_factor
        return self.propagator.init_output_distribution(
            output_logits, target_class, true_class
        )

    def check_spatial_integrity(self, dist_in, a_prev, layer_name, node_type, tolerance=1e-7):
        """
        Checks if probability is being attributed to neurons that were inactive 
        during the forward pass. Delegate to propagator.
        """
        return self.propagator.check_spatial_integrity(
            dist_in, a_prev, layer_name, node_type, tolerance
        )

    # ------------------------------------------------------------------
    # Handler methods - delegate to propagator
    # ------------------------------------------------------------------
    
    def _handle_linear(self, ctx: dict) -> tuple:
        a_curr = ctx['a_curr']
        if a_curr is None:
            _, a_raw = self._prev_activation(ctx['prev_layer'])[0]
            a_curr = a_raw.view(-1)
        return self.propagator.paf_propagate_linear(
            dist_out       = ctx['dist_out'],
            a_in           = a_curr,
            w              = ctx['W'],
            score_fn        = ctx['score_fn'],
            mode_key        = ctx['mode_key']
        ), a_curr

    def _handle_conv(self, ctx):
        return self.propagator.paf_propagate_conv(
            dist_out       = ctx['dist_out'],
            a_in           = ctx['a_curr'],
            w              = ctx['W'],
            score_fn        = ctx['score_fn'],
            layer          = ctx['layer'],
            mode_key        = ctx['mode_key'],
            cache_key       = ctx['cache_key'],
        ), ctx['a_curr']

    def _handle_maxpool(self, ctx):
        return self.propagator.paf_propagate_maxpool(
            a_in           = ctx['a_curr'],
            a_out          = self.activations.get(ctx['curr_layer']),
            dist_out       = ctx['dist_out'],
            layer          = ctx['layer'],
            mode_key        = ctx['mode_key']
        ), ctx['a_curr']

    def _handle_avgpool(self, ctx):
        return self.propagator.paf_propagate_avgpool(
            a_in           = ctx['a_curr'],
            dist_out       = ctx['dist_out'],
            score_fn        = ctx['score_fn'],
            layer          = ctx['layer'],
            mode_key        = ctx['mode_key']
        ), ctx['a_curr'].abs()

    def _handle_flatten(self, ctx):
        a_curr = ctx['a_curr']
        dist_in = ctx['dist_out'].view(a_curr.shape)
        return dist_in, a_curr
    
    def _handle_passthrough(self, ctx: dict) -> tuple:
        """relu, batchnorm, layernorm, gelu, dropout — distribution unchanged."""
        dist   = ctx['dist_out']
        a_curr = ctx['a_curr']
        # Align shape to a_curr — handles ViT rank mismatches
        if a_curr is None:
            return torch.zeros_like(dist), None
        if a_curr is not None and dist.shape != a_curr.shape:
            try:
                dist = dist.reshape(a_curr.shape)
            except RuntimeError:
                if dist.dim() == a_curr.dim() + 1 and dist.shape[0] == 1:
                    dist = dist.squeeze(0)
                elif dist.dim() + 1 == a_curr.dim():
                    dist = dist.unsqueeze(0)
        return dist, a_curr
    
    def _handle_reshape(self, ctx: dict) -> tuple:
        """reshape, permute, view, contiguous — reshape dist to match input."""
        dist   = ctx['dist_out']
        a_curr = ctx['a_curr']
        if a_curr is None:
            return dist, a_curr
        try:
            dist = dist.reshape(a_curr.shape)
        except RuntimeError:
            if dist.dim() == a_curr.dim() + 1 and dist.shape[0] == 1:
                dist = dist.squeeze(0)
            elif dist.dim() + 1 == a_curr.dim():
                dist = dist.unsqueeze(0)
        return dist, a_curr
    
    def _handle_mhsa(self, ctx):
        return self.propagator.paf_propagate_mhsa(
            dist_out = ctx['dist_out'],
            a_in     = ctx['a_curr'],
            layer    = ctx['layer'],
            score_fn = ctx['score_fn'],
        ), ctx['a_curr']

    def _handle_cls_token(self, ctx: dict) -> tuple:
        """
        Reverse of x[:, 0] — routes all mass to token position 0.
        """
        return self.propagator.paf_propagate_cls_token(
            ctx['dist_out'], ctx['a_curr']
        ), ctx['a_curr']

    def _handle_cat_token(self, ctx: dict) -> tuple:
        """
        Reverse of torch.cat([class_token, patch_tokens], dim=1).
        Drops CLS column — it came from a Parameter not input image.
        """   
        return self.propagator.paf_propagate_cat_node(
            curr_layer=ctx['curr_layer'],
            dist_out=ctx['dist_out'],
            pred_pairs=ctx['pred_act_list'],
            distributions=ctx['distributions'],
            has_skip=ctx['has_skip'],
            mode_key=ctx['mode_key'],
            score_fn=ctx['score_fn'],
            store_edge_fn = self._store_edge,
        ), None

    def _handle_parameter(self, ctx: dict) -> tuple:
        """Leaf parameter node — no predecessors, mass absorbed here."""
        return torch.zeros_like(ctx['dist_out']), None

    def _handle_join(self, ctx: dict) -> tuple:
        """
        Join node — multiple predecessors but only one carries tensor signal.
        Examples: expand(tensor, scalar), reshape(tensor, shape_tuple)

        Pass full distribution to tensor predecessor unchanged.
        Scalar predecessors (shape args) receive nothing — they carry
        no attribution signal.
        """
        pred_act_list = ctx['pred_act_list']
        for (pred, act) in pred_act_list:
            ctx['prev_layer'] = pred
            ctx['a_curr'] = act
            dist, _ = self._handle_passthrough(ctx)
            self._store_edge(
                prev_layer=pred, curr_layer=ctx['curr_layer'],
                dist=dist,
                distributions=ctx['distributions'], 
                skip=ctx['has_skip'],
                mode_key=ctx['mode_key']    
            )
        return None, None

    def _handle_add(self, ctx: dict) -> tuple:
        """
        ADD node — handled separately since it returns dict not tensor.
        """
        dist_map = self._process_add_node(
            curr_layer     = ctx['curr_layer'],
            dist_out       = ctx['dist_out'],
            pred_pairs     = ctx['pred_act_list'], 
            distributions  = ctx['distributions'],
            has_skip       = ctx['has_skip'],
            score_fn       = ctx['score_fn'],
            mode_key       = ctx['mode_key']
        )
        # Store results
        for name, dist in dist_map.items():
            self._store_edge(
                prev_layer    = name,
                curr_layer    = ctx['curr_layer'],
                dist          = dist,
                distributions = ctx['distributions'],
                skip          = ctx['has_skip'],
                mode_key      = ctx['mode_key'],
            )

        if self.debug_level:
            self._check_add_mass(ctx['curr_layer'],
                                [n for n, _ in dist_map.items()],
                                ctx['dist_out'], ctx['mode_key'])

        return None, None   # signals explain to use _process_add_node

    # ------------------------------------------------------------------
    # Distribution storage helpers
    # ------------------------------------------------------------------
    @profile 
    def _store_edge(
        self,
        prev_layer: str,
        curr_layer: str,
        dist: torch.Tensor,
        distributions: Dict[str, torch.Tensor],
        mode_key:      tuple, 
        skip: bool = False,
    ) -> None:
        """
        Store distributions for prev_layer and record edge mass.
        """
        dist = dist.detach()  
        existing = distributions.get(prev_layer)
        if (skip and existing is not None):
            existing.add_(dist)             # in-place addition
        else:
            distributions[prev_layer] = dist
        self.edge_mass[mode_key][(prev_layer, curr_layer)] = dist.sum().item()
        
    @profile 
    def _get_dist(
        self,
        layer: str,
        distributions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Retrieve distributions for a layer.
        """
        if layer not in distributions:
            raise KeyError(
                f"Layer '{layer}' not found in distributions. "
                f"Available layers: {list(distributions.keys())[:5]}. "
                f"Check backward_order and graph traversal."
            )
        return distributions[layer]
        
    @profile 
    def _prev_activation(
        self,
        layer_name: str,
    ) -> List[Tuple[str, Optional[torch.Tensor]]]:
        """
        Return (predecessor_name, predecessor_activation).
        """
        preds = self.graph_manager.graph_info.predecessors.get(layer_name, [])
        if not preds:
            return [("no_predecessor", None)]
        
        return [(pred, self.activations.get(pred)
               if isinstance(self.activations.get(pred), torch.Tensor)
               else None) for pred in preds]

    def _process_add_node(
        self,
        curr_layer:    str,
        dist_out:      torch.Tensor,
        pred_pairs:    List[Tuple[str, Optional[torch.Tensor]]],
        distributions: Dict[str, torch.Tensor],
        has_skip:      bool,
        mode_key:      tuple,
        score_fn:      callable,
    ) -> Dict[str, torch.Tensor]:
        """
        PAF backward through additive merge.
        Delegate to propagator.
        """
        return self.propagator.paf_propagate_add(
            preds=[pred for pred, _ in pred_pairs],
            dist_out=dist_out,
            act_preds=[act for _, act in pred_pairs],
            score_fn=score_fn
        )

    def _check_add_mass(
        self,
        curr_layer: str,
        preds: List[str],
        dist_out: torch.Tensor,
        mode_key: tuple,
    ) -> None:
        """Verify mass conservation at an ADD node for a specific mode."""
        total = sum(
            self.edge_mass[mode_key].get((p, curr_layer), 0.0)
            for p in preds
        )
        loss = abs(total - dist_out.sum().item())
        if loss > 1e-4:
            print(
                f"ADD mass loss at {curr_layer} "
                f"(mode={mode_key[0].value}, tau={mode_key[1]}): "
                f"out={dist_out.sum():.4f}, "
                f"distributed={total:.4f}"
            )

    # ------------------------------------------------------------------
    # Main explain method
    # ------------------------------------------------------------------
    
    @profile 
    def explain(self, weights, target_class, true_class=None):
        """
        Main explanation method that orchestrates the backward propagation.
        """
        # One distribution dict per mode
        distributions = {key: {} for key in self._score_fns}

        backward_order = self.graph_manager.graph_info.backward_order
        successors = self.graph_manager.graph_info.successors

        # Initialise output layer
        output_layer = backward_order[0]
        out_dist = self.init_output_distribution(
            self.activations[output_layer],
            target_class=target_class,
            true_class=true_class
        )
        for key in self._score_fns:
            distributions[key][output_layer] = out_dist.clone()
       
        # Backward traversal
        for i in range(len(backward_order) - 1):
            curr_layer = backward_order[i]
            node_type = self.graph_manager.graph_info.node_types.get(curr_layer, 'unknown')
            pred_act_list = self._prev_activation(curr_layer)
            prev_layer, a_curr = pred_act_list[0]    # Most layer has single predecessor
            has_skip = len(successors.get(prev_layer, [])) > 1

            if pred_act_list == [('no_predecessor', None)]:
                continue

            handler = self._dispatch.get(node_type, self._handle_passthrough)
            a_for_check = None

            for key in self._score_fns:
                dist_out = self._get_dist(
                    curr_layer,
                    distributions[key],
                )

                ctx = {
                    'curr_layer'      : curr_layer,
                    'prev_layer'      : prev_layer,
                    'a_curr'          : a_curr,
                    'dist_out'        : dist_out,
                    'layer'           : self.graph_manager.graph_info.module_map.get(curr_layer),
                    'W'               : weights.get(curr_layer, None),
                    'score_fn'        : self._score_fns[key],
                    'mode_key'        : key,
                    'cache_key'       : curr_layer,
                    'has_skip'        : has_skip,
                    'distributions'   : distributions[key],
                    'pred_act_list'   : pred_act_list
                }

                dist_in, a_for_check = handler(ctx)

                # distributions are stored inside handler
                if node_type in {'add', 'cat', 'join'}:   
                    continue

                self._store_edge(
                    prev_layer=prev_layer, curr_layer=curr_layer,
                    dist=dist_in,
                    distributions=distributions[key], skip=has_skip,
                    mode_key=key    
                )
            '''
            if node_type == 'conv':
                values = []
                for key in self._score_fns:
                    d = distributions[key].get(prev_layer)
                    if d is not None:
                        values.append((key[0].value, d.sum().item(), d.mean().item()))
                if len(values) > 1:
                    sums = [v[1] for v in values]
                    if abs(max(sums) - min(sums)) < 1e-6:
                        print(f"IDENTICAL at {curr_layer} ({node_type}): {values}")
                    else:
                        print(f"DIVERGED  at {curr_layer} ({node_type}): {values}")
                        

            if node_type == 'conv':
                keys = list(self._score_fns.keys())
                # Print all pairwise comparisons including non-consecutive
                for i, k1 in enumerate(keys):
                    for k2 in keys[i+1:]:
                        d1 = distributions[k1].get(prev_layer)
                        d2 = distributions[k2].get(prev_layer)
                        if d1 is not None and d2 is not None:
                            l2  = (d1.flatten() - d2.flatten()).norm().item()
                            cos = torch.nn.functional.cosine_similarity(
                                d1.flatten().unsqueeze(0),
                                d2.flatten().unsqueeze(0)
                            ).item()
                            print(f"  {k1[0].value} vs {k2[0].value}: "
                                f"L2={l2:.6f}, cos={cos:.8f}, "
                                f"same_obj={d1 is d2}")
            '''  
            # Clear unfold cache from propagator
            if hasattr(self.propagator, '_unfold_cache'):
                self.propagator._unfold_cache.clear()

            if a_for_check is not None:
                self.check_spatial_integrity(
                    dist_in, a_for_check, curr_layer, node_type
                )

        # Store and optionally scale
        self.distributions = distributions

        if self.scaling_factor != 1.0:
            scale = self.scaling_factor
            self.distributions = {
                key: {k: v * scale for k, v in d.items()}
                for key, d in distributions.items()
            }
            self.edge_mass = {
                key: {k: v * scale for k, v in d.items()}
                for key, d in self.edge_mass.items()
            }
        return distributions

    def extract_weights(
        self,
        model: nn.Module,
        conv_reshape: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract weight matrices for any PyTorch model architecture.
        """
        weights: Dict[str, torch.Tensor] = {}
        traced_graph = self.graph_manager.graph_info.traced.graph

        for node in traced_graph.nodes:
            if node.op != 'call_module':
                continue

            name = node.name 
            module = model.get_submodule(node.target)

            if not hasattr(module, "weight") or module.weight is None:
                continue

            w = module.weight.detach().clone()   # preserve device/dtype, no gradient

            # 1. Linear Layers: (Out, In) -> (In, Out) 
            if isinstance(module, nn.Linear):
                weights[name] = w.T
                
            # 2. Conv Layers: KEEP ORIGINAL (Out, In, K, K)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                weights[name] = w 
                
            # 3. Everything else (Embeddings, etc.)
            else:
                weights[name] = w

        return weights

    def cleanup(self):
        """Clean up memory and caches."""
        # Clear activations
        if hasattr(self, 'activations'):
            self.activations.clear()
        
        # Clear distributions and edge mass
        for key in list(self.distributions.keys()):
            self.distributions[key].clear()
        for key in list(self.edge_mass.keys()):
            self.edge_mass[key].clear()
        
        # Clear unfold cache from propagator
        if hasattr(self.propagator, '_unfold_cache'):
            self.propagator._unfold_cache.clear()
        
        # Clear any lingering large tensors in score functions if needed
        if hasattr(self, '_score_fns'):
            self._score_fns.clear()
    
        # Force garbage collection + CUDA cache
        torch.cuda.empty_cache()
    
    def run(
        self,
        x:              torch.Tensor,
        target_label:   int,
        output_mode:    str = None,
        true_label:     Optional[int] = None,
    ):
        """
        Entry function to run PAF on a single input sample. 
        """
        x = x.to(self.device)
        if output_mode is not None:
            self.output_mode = output_mode
            
        with torch.no_grad():
            logits = self.model(x)

        self.graph_manager.run_forward(x)
        activations = {k: v for k, v in self.graph_manager.activations.items()}
        activations["input"] = x
        self.activations = activations
        weights = self.extract_weights(self.graph_manager.model)
        self.explain(weights=weights, target_class=target_label, true_class=true_label)

    # Utility methods (keeping original implementations)
    def dominant_path(
        self, distributions: Dict[str, torch.Tensor], layer_order: List[str]
    ) -> Dict[str, int]:
        """Extract dominant inference path: argmax neuron at each layer."""
        path = {}
        for layer in layer_order:
            if layer not in distributions:
                continue
            dist = distributions[layer]
            if dist.dim() == 1:
                path[layer] = int(dist.argmax())
            elif dist.dim() == 3:
                C, H, W = dist.shape
                flat_idx = dist.argmax()
                c = flat_idx // (H*W)
                h = (flat_idx % (H*W)) // W
                w = flat_idx % W
                path[layer] = (c, h, w)
            else:
                raise ValueError(f"Unsupported distribution shape {dist.shape}")
        return path

    def find_layers_with_types(self, mode, layer_type, tolerance=0.0005):
        """Find layers of specific type."""
        layers_to_find = []
        
        for layer_name, tensor in self.distributions[mode].items():
            nodetype = self.graph_manager.graph_info.node_types.get(layer_name, 'unknown')
            
            if layer_type == nodetype:
                layers_to_find.append(layer_name) 
        return layers_to_find
    
    def layer_entropy(
        self, distributions: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute Shannon entropy of probability distribution at each layer."""
        entropies = {}
        for layer, p in distributions.items():
            p_safe = p.clamp(min=1e-12)
            entropies[layer] = float(-(p_safe * p_safe.log()).sum())
        return entropies
