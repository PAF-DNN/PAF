"""
Layer Propagation Engine for PAF
==================================
Extracts layer-specific propagation methods from PAF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import math
from itertools import product
from torch.nn.modules.conv import _ConvNd
from core.paf.graph.tracer import GraphInfo

class LayerPropagator:
    """
    Layer-specific propagation methods extracted from PAF.
    
    This class contains all helper methods and layer propagation methods
    for computing backward probability distributions through neural network layers.
    
    Note: The engine attribute (containing distribute_attribution method)
    must be set after instantiation.
    """
    
    # Class-level pooling mappings
    POOL_MAP = {
        nn.AdaptiveAvgPool1d: 1, nn.AvgPool1d: 1,
        nn.AdaptiveAvgPool2d: 2, nn.AvgPool2d: 2,
        nn.AdaptiveAvgPool3d: 3, nn.AvgPool3d: 3,
        # MaxPool
        nn.MaxPool1d:         1,
        nn.MaxPool2d:         2,
        nn.MaxPool3d:         3,
        nn.AdaptiveMaxPool1d: 1,
        nn.AdaptiveMaxPool2d: 2,
        nn.AdaptiveMaxPool3d: 3,
    }
    
    UNPOOL_MAP = {
        1: (F.max_pool1d, F.max_unpool1d),
        2: (F.max_pool2d, F.max_unpool2d),
        3: (F.max_pool3d, F.max_unpool3d),
    }
    
    def __init__(self, graph_info: GraphInfo, debug_level: int = 0, eps: float = 1e-9, model: Optional[nn.Module] = None):
        """
        Initialize the LayerPropagator.
        
        Parameters
        ----------
        graph_info : PAFGraphInfo
            Information about the graph structure
        debug_level : int
            Debug verbosity level (0=silent, higher=more verbose)
        eps : float
            Numerical stability constant
        model : torch.nn.Module, optional
            Reference to the model being analyzed
        """
        self.debug_level = debug_level
        self.eps = eps
        self.model = model
        self.graph_info = graph_info
        self.engine = None  # Must be set after instantiation

        # Cache for unfold operations
        self._unfold_cache: Dict[str, torch.Tensor] = {}
    
    def set_engine(self, engine):
        """Set the attribution engine after instantiation."""
        self.engine = engine

    def check_spatial_integrity(self,dist_in, a_prev, layer_name, node_type, tolerance=1e-7):
        """
        Checks if probability is being attributed to neurons that were inactive 
        during the forward pass.
        """
        if self.debug_level < 2:
            return -1.0
        # Skip check for these layer types (they don't have "dead" neurons)
        skip_layers = {
                'batchnorm', 'layernorm', 'groupnorm',
                'instancenorm', 'dropout', 'passthrough',
                'reshape', 'permute', 'join',
            }        

        if node_type in skip_layers:
            #print(f"✅ [Layer: {layer_name:15}] | Skipped (no dead neuron check for {node_type})")
            return 0.0
        with torch.no_grad():
            # 1. Identify where activations were non-positive (dead neurons)
            if node_type == 'relu':
                dead_mask = (a_prev <= tolerance)
            else:
                dead_mask = (a_prev.abs() <= tolerance)  

            # Align shapes if needed
            if dist_in.shape != a_prev.shape:
                try:
                    dead_mask = dead_mask.reshape(dist_in.shape)
                except RuntimeError:
                    return -1.0

            leakage    = dist_in[dead_mask].abs().sum().item()
            total_p    = dist_in.abs().sum().item()
            leak_ratio = leakage / total_p if total_p > 0 else 0.0


            # 3. Calculate total probability to get a relative sense of the leak
            if leakage > tolerance:
                print(f"⚠️ Layer: {layer_name} | "
                    f"dist: {dist_in.sum().item():.4f} | "
                    f"leak: {leakage:.4f} | "
                    f"leak%: {leak_ratio*100:5.2f}% | "
                    f"dead: {dead_mask.sum().item():,d}")
                #print(f"⚠️ [Layer: {layer_name:15}, dist_sum: {dist_in.sum().item():.6f} ] | Leakage: {leakage:.6e} | "
                #      f"Leak %: {leak_ratio*100:6.2f}% | Dead Pixels: {dead_mask.sum().item()}")
            elif self.debug_level >= 3:
                print(
                    f"✅ [{layer_name:40s}] ({node_type:12s}) "
                    f"no leak"
                )

        return leakage

    def init_output_distribution(
        self,
        output_logits: torch.Tensor,
        target_class: Optional[int] = None,
        true_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Initialise the probability distribution at the output layer.

        Inputs:
            output_logits : raw output values (logits or probabilities)
            mode          : "target"  -> one-hot at target_class (Option 1)
                            "softmax" -> softmax of logits (Option 2)
                            "normalise" -> normalise positive logits (Option 2 variant)
            target_class  : required when mode="target"
        Returns:
            p_out : valid probability distribution over output neurons
        """
        K = output_logits.shape[-1]
        batch_size = output_logits.shape[:-1]

        if self.output_mode == "target":
            assert target_class is not None, "target_class required for mode='target'"
            p_out = torch.zeros_like(output_logits)
            p_out[...,target_class] = 1.0
            return p_out
        elif self.output_mode == "softmax":
            return F.softmax(output_logits, dim=-1)
        elif self.output_mode == "softmax_target_only":   # ← new mode
            assert target_class is not None, "target_class required for mode='softmax_target_only'"
            probs = F.softmax(output_logits, dim=-1)           # normal softmax
            p_out = torch.zeros_like(probs)
            p_out[...,target_class] = probs[...,target_class]         # keep only the prob of target
            return p_out

        elif self.output_mode == "normalise":
            # Lift to positive then normalise — avoids softmax temperature sensitivity
            lifted, _ = self.lift(output_logits, eps=1e-6)
            lift_sum = lifted.sum()
            lift_max = torch.clamp(lift_sum, min=1e-6)
            return lifted / lift_max
        elif self.output_mode == "contrastive_explanation_predicted":  
            assert target_class is not None and true_class is not None, "Both classes required"

            # 1. Get raw probabilities
            probs = F.softmax(output_logits, dim=-1)
            p_pred = probs[..., target_class]
            p_true = probs[..., true_class]

            # 2. PAF makes the sum of mass 1 in all layers
            # scaling_factor transforms mass by the factor of output probability
            total_pair_prob = p_pred + p_true
            norm_p_pred = p_pred / total_pair_prob

            # 3. Create the output tensor
            p_out = torch.zeros_like(probs)
            p_out[..., target_class] = norm_p_pred
            self.scaling_factor = norm_p_pred.sum().item() # Final mass for later comparison
            return p_out
        elif self.output_mode == "contrastive_explanation_true":  
            assert target_class is not None and true_class is not None, "Both classes required"

            # 1. Get raw probabilities
            probs = F.softmax(output_logits, dim=-1)
            p_pred = probs[..., target_class]
            p_true = probs[..., true_class]

            # 2. Re-normalize to a sum of 1 between the pair
            total_pair_prob = p_pred + p_true
            norm_p_true = p_true / total_pair_prob

            # 3. Create the output tensor
            p_out = torch.zeros_like(probs)
            p_out[..., true_class] = norm_p_true
            self.scaling_factor = norm_p_true.sum().item() # Final mass for later comparison
            return p_out
        else:
            raise ValueError(f"mode must be 'target', 'softmax', or 'normalise', got {self.output_mode}")

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lift each sample in a batch independently to be strictly non-negative.
        Inputs:
            x   : (N, ...) tensor. 

        Returns:
            lifted    : Tensor of same shape as x.
            triggered : (N,) boolean tensor indicating which samples were lifted.
        """
        # 1. Determine the 'Sample' dimensions (everything after Batch)
        # If x is (N, C, H, W), we reduce over (1, 2, 3)
        reduce_dims = list(range(1, x.dim()))

        # 2. Compute minimum per sample
        # min_vals shape: (N, 1, 1, 1...) for easy broadcasting
        min_vals = x
        for dim in reduce_dims:
            min_vals = min_vals.min(dim=dim, keepdim=True).values

        # 3. Identify which samples in the batch need lifting
        triggered = min_vals < 0  # Shape (N, 1, 1...)

        # 4. Apply lifting only to samples where min < 0
        lifted = torch.where(triggered, x - min_vals, x)

        # Return triggered as a flat boolean mask (N,)
        return lifted, triggered.view(-1)


    def process_activations(self, a: torch.Tensor, delta: int) -> torch.Tensor:
        """
        Apply activation treatment according to delta.

        delta=0 : raw (may be negative)
        delta=1 : conditional lifting
        """
        if delta == 0:
            return a
        elif delta == 1:
            lifted, _ = self.lift(a)
            return lifted
        else:
            raise ValueError(f"delta must be 0 or 1, got {delta}")


    def process_weights(self,w: torch.Tensor, gamma: int) -> torch.Tensor:
        """
        Apply weight treatment according to gamma.
            gamma=0 : signed (raw)
            gamma=1 : conditional lifting
            gamma=2 : absolute value
        """
        if gamma == 0:
            return w
        elif gamma == 1:
            lifted, _ = self.lift(w)
            return lifted
        elif gamma == 2:
            return w.abs()
        else:
            raise ValueError(f"gamma must be 0, 1, or 2, got {gamma}")


    def paf_propagate_conv(
        self,
        dist_out:        torch.Tensor,
        a_in:            torch.Tensor,
        w:               torch.Tensor,
        layer:           _ConvNd,
        score_fn:        callable,             
        mode_key:         Tuple,
        cache_key:  str = None
    ) -> torch.Tensor:

        # 1. Dynamic Geometry Extraction
        # This works whether kernel is an int (3) or a tuple (3, 3)

        if a_in.dim() == len(layer.kernel_size) + 1:
            a_in = a_in.unsqueeze(0)
        if dist_out.dim() == len(layer.kernel_size) + 1:
            dist_out = dist_out.unsqueeze(0)

        k = layer.kernel_size
        s = layer.stride
        p = layer.padding
        d = layer.dilation

        N = a_in.shape[0]
        C_out = w.shape[0]
        # spatial_in: captures (H, W) or (D, H, W) or (L)


        if cache_key and cache_key in self._unfold_cache:
            a_patches = self._unfold_cache[cache_key]
            is_cached = False
        else:
            a_patches = F.unfold(a_in, kernel_size=k, stride=s, padding=p, dilation=d)
            if cache_key:
                self._unfold_cache[cache_key] = a_patches
            is_cached = True

        L = a_patches.shape[-1]
        w_flat = w.view(C_out, -1) # Flatten [C_in, k, k] -> [patch_size]

        spatial_in = list(a_in.shape[2:]) 

        # Generate mask dynamically based on the input shape
        #mask = F.unfold(torch.ones_like(a_in), kernel_size=k, stride=s, padding=p, dilation=d)

        # 3. The Math (The 'Logic' Step)
        # We pass the flattened tensors to the engine

        #if L > 1024:
        p_in_unfolded = self.engine.distribute_attribution(
            dist_out=dist_out.view(N, C_out, 1, -1),
            a_patches=a_patches.unsqueeze(1),
            w_flat=w_flat.unsqueeze(0).unsqueeze(-1),
            score_fn=score_fn,
        )

        # 4. Map back to Input Space
        dist_in = F.fold(p_in_unfolded, output_size=spatial_in, kernel_size=k, stride=s, padding=p, dilation=d)
        #dist_in = self._layer_normalise(dist_in, mode_key)
        
        #if is_cached:
        #    del a_patches
        #del mask
        return dist_in

    def paf_propagate_linear(
        self,
        dist_out:        torch.Tensor,
        a_in:            torch.Tensor,
        w:               torch.Tensor,
        score_fn:        callable,              
        mode_key:         Tuple
    ) -> torch.Tensor:
        """
        Truly General Linear Wrapper.
        p_out: (N, ..., K)  <-- Attribution from classes (1000)
        a_in:  (N, ..., J)  <-- Activations from features (512)
        w:     (K, J)       <-- Weights (1000, 512)
        """
        # 1. Capture exact feature counts
        # We look at the last dimension of the tensors
        K = dist_out.shape[-1] 
        J = a_in.shape[-1]

        # 2. Flatten leading dimensions (N, T, etc.) into one Batch dimension
        # This handles [N, 1000] or [N, T, 1000] identically
        p_flat = dist_out.reshape(-1, K)
        a_flat = a_in.reshape(-1, J)

        N_total = a_flat.shape[0]

        # 3. Standardize for Engine (N, K, J, L)
        # Using reshape() here prevents the "contiguous subspace" crash
        p_standard = p_flat.reshape(N_total, K, 1, 1)
        a_standard = a_flat.reshape(N_total, 1, J, 1)
        w_standard = w.reshape(1, K, J, 1)

        # 4. Call the Engine (Universal Math)
        p_in_engine= self.engine.distribute_attribution(
            dist_out=p_standard,
            a_patches=a_standard,
            w_flat=w_standard,
            score_fn=score_fn
        )

        # 5. Restore original input shape (N, ..., J)
        # The engine returns (N_total, J, 1) -> view back to (N, 512)
        p_in = p_in_engine.view(a_in.shape)
        #p_in = self._layer_normalise(p_in, mode_key)

        return p_in

    def align_dims(self, p_tensor:torch.Tensor, reference_tensor:torch.Tensor) -> torch.Tensor:
        """
        Dynamically aligns p_tensor's rank to match reference_tensor.
            - If p_tensor is missing leading dimensions (e.g., Batch), unsqueeze at the front.
        """
        # Calculate how many dimensions are missing
        diff = reference_tensor.dim() - p_tensor.dim()

        # If the probability is missing dimensions (like the Batch dim),
        # unsqueeze from the front until they match.
        for _ in range(diff):
            p_tensor = p_tensor.unsqueeze(0)

        return p_tensor
    def paf_propagate_maxpool(
            self, 
            a_in:torch.Tensor, 
            a_out:torch.Tensor, 
            dist_out:torch.Tensor, 
            layer:torch.nn.modules.pooling._MaxPoolNd,
            mode_key: Tuple = None
    ) -> torch.Tensor:
        """
        MaxPool backward propagation: winner-takes-all approach.
        The neuron(s) in the input patch that had the maximum activation during the forward pass receive all the probability mass from the output neuron.
        Inputs:
            a_curr: activation of previous layer (input to MaxPool)
            a_out: activation of current layer (output of MaxPool)
            dist_out: probability distribution at the output of MaxPool
            layer: the MaxPool layer object (to access kernel_size, stride, padding, dilation)
        """
        # Implementation would require tracking max indices during forward pass
        # Ensure inputs have batch dimension

        ndim = self.POOL_MAP.get(type(layer))
        if ndim is None:
            raise ValueError(f"Unsupported maxpool layer type: {type(layer)}")
        expected_dim = ndim + 2  # N + C + spatial dims
        # Align tensor ranks consistently
        a_in_nd     = a_in.unsqueeze(0)     if a_in.dim()     < expected_dim else a_in
        a_out_nd    = a_out.unsqueeze(0)    if a_out.dim()     < expected_dim else a_out
        dist_out_nd = dist_out.unsqueeze(0) if dist_out.dim()  < expected_dim else dist_out

        f_pool, f_unpool = self.UNPOOL_MAP[ndim]

        # Check dilation — f_unpool does not support dilation != 1
        dilation = getattr(layer, 'dilation', 1)
        dilation_vals = dilation if isinstance(dilation, (tuple, list)) else [dilation]
        if any(d != 1 for d in dilation_vals):
            raise NotImplementedError(
                f"MaxPool with dilation={dilation} is not supported by "
                f"F.max_unpool. PAF cannot propagate through this layer exactly."
            )

            # Recompute pool indices from stored input activations
        _, pool_indices = f_pool(
            a_in_nd,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=getattr(layer, 'dilation', 1),
            ceil_mode=getattr(layer, 'ceil_mode', False),
            return_indices=True
        )

        # Validate shapes
        if pool_indices.shape != dist_out_nd.shape:
            raise ValueError(
                f"MaxPool shape mismatch: "
                f"pool_indices={pool_indices.shape}, "
                f"dist_out={dist_out_nd.shape}. "
                f"Check activation storage in graph traversal."
            )

        # Winner-takes-all routing
        N, C       = a_in_nd.shape[:2]
        spatial_in = a_in_nd.shape[2:]
        flat_in    = 1
        for s in spatial_in:
            flat_in *= s  # H_in * W_in (and D_in for 3D)

        indices_flat  = pool_indices.reshape(N, C, -1)       # (N, C, H_out*W_out)
        dist_out_flat = dist_out_nd.reshape(N, C, -1)        # (N, C, H_out*W_out)

        # Allocate output and accumulate
        dist_in_flat = torch.zeros(N, C, flat_in,
                                device=dist_out.device,
                                dtype=dist_out.dtype)
        dist_in_flat.scatter_add_(
            dim=2,
            index=indices_flat,
            src=dist_out_flat
        )
        dist_in = dist_in_flat.reshape(a_in_nd.shape)

        # Mass conservation — must hold exactly
        mass_loss = (dist_out_nd.sum() - dist_in.sum()).abs().item()
        if mass_loss > 1e-6:
            print(
                f"WARNING MaxPool mass loss = {mass_loss:.8f} "
                f"(out={dist_out_nd.sum():.6f}, in={dist_in.sum():.6f}) "
                f"layer={type(layer).__name__}, "
                f"a_in={a_in_nd.shape}, dist_out={dist_out_nd.shape}"
            )

        return dist_in.reshape(a_in.shape)
    def paf_propagate_avgpool(
        self,
        dist_out: torch.Tensor,
        a_in: torch.Tensor,
        score_fn:        callable,
        layer: nn.Module,
        mode_key:         Tuple
    ) -> torch.Tensor:

        ndim = self.POOL_MAP.get(type(layer))
        if ndim is None:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
        expected_dim = ndim + 2  # N + C + spatial dims

        # Align tensor ranks
        a_in_nd  = a_in.unsqueeze(0)  if a_in.dim()  < expected_dim else a_in
        p_out_nd = dist_out.unsqueeze(0) if dist_out.dim() < expected_dim else dist_out

        N, C     = a_in_nd.shape[:2]
        spatial_in = a_in_nd.shape[2:]

        def to_tuple(val):
            return val if isinstance(val, tuple) else (val,) * ndim

        # --- Determine kernel parameters ---
        if isinstance(layer, (
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d
        )):
            spatial_out = p_out_nd.shape[2:]

            # Check exact divisibility
            exact = all(si % so == 0 for si, so in zip(spatial_in, spatial_out))
            if not exact:
                # Fall back to exact receptive field inverse
                return self.adaptive_avgpool_exact_inverse(
                    a_in=a_in_nd,
                    dist_out=p_out_nd,
                    score_fn=score_fn,
                    spatial_start_dim=2
                )

            k_tuple = tuple(si // so for si, so in zip(spatial_in, spatial_out))
            s_tuple = k_tuple
            p_tuple = (0,) * ndim

        else:
            k_tuple = to_tuple(layer.kernel_size)
            s_tuple = to_tuple(
                layer.stride if layer.stride is not None else layer.kernel_size
            )
            p_tuple = to_tuple(layer.padding)

        # --- Shape verification ---
        for d in range(ndim):
            si, k, s, p = spatial_in[d], k_tuple[d], s_tuple[d], p_tuple[d]
            expected_out = (si + 2*p - k) // s + 1
            actual_out   = p_out_nd.shape[d + 2]
            if expected_out != actual_out:
                raise ValueError(
                    f"AvgPool shape mismatch at spatial dim {d}: "
                    f"expected {expected_out}, got {actual_out}. "
                    f"(in={si}, k={k}, s={s}, p={p})"
                )

        # --- Process each channel independently via reshape ---
        # AvgPool is depthwise — channel c output depends only on channel c input
        # Reshape (N, C, *spatial) → (N*C, 1, *spatial) to process per channel
        # This avoids the groups mismatch issue entirely
        NC = N * C
        a_reshaped   = a_in_nd.reshape(NC, 1, *spatial_in)
        p_reshaped   = p_out_nd.reshape(NC, 1, *p_out_nd.shape[2:])

        # Unfold each (NC, 1, *spatial) independently
        # patch_size = 1 * k_h * k_w (C_in=1 per group)
        a_patches = F.unfold(
            a_reshaped,
            kernel_size=k_tuple, stride=s_tuple, padding=p_tuple
        )  # (NC, patch_size, L),  patch_size = k_h * k_w

        patch_size = a_patches.shape[1]  # = k_h * k_w (NOT C * k_h * k_w)
        L          = a_patches.shape[2]

        # Reshape to _distribute_attribution format
        # a_standard: (NC, 1, patch_size, L)
        # w_standard: (1,  1, patch_size, 1) — unit weights
        # dist_out:   (NC, 1, 1,          L)
        a_standard    = a_patches.unsqueeze(1)           # (NC, 1, patch_size, L)
        w_standard    = torch.ones(
            1, 1, patch_size, 1,
            device=a_in.device, dtype=a_in.dtype
        )
        dist_standard = p_reshaped.reshape(NC, 1, 1, L)  # (NC, 1, 1, L)
        # --- Delegate to attribution engine ---
        p_in_unfolded= self.engine.distribute_attribution(
            dist_out        = dist_standard,
            a_patches       = a_standard,
            w_flat          = w_standard,
            score_fn=score_fn
        )
        # p_in_unfolded: (NC, patch_size, L)

        # --- Fold back ---
        p_in_folded = F.fold(
            p_in_unfolded,
            output_size=list(spatial_in),
            kernel_size=k_tuple, stride=s_tuple, padding=p_tuple
        )  # (NC, 1, *spatial_in)

        # --- Reshape back to (N, C, *spatial_in) ---
        p_in_4d      = p_in_folded.reshape(N, C, *spatial_in)

        # --- Mass conservation check ---
        mass_loss = (p_out_nd.sum() - p_in_4d.sum()).abs().item()
        if mass_loss > 1e-3:
            print(f"WARNING avgpool mass loss = {mass_loss:.6f} "
                f"(out={p_out_nd.sum():.4f}, in={p_in_4d.sum():.4f})")

        return p_in_4d.reshape(a_in.shape)

    def adaptive_avgpool_exact_inverse(
        self,
        a_in: torch.Tensor,
        dist_out: torch.Tensor,
        score_fn: callable, 
        mode_key:         Tuple,
        spatial_start_dim: int = 2,
    ) -> torch.Tensor:
        """
        Exact PAF inverse for AdaptiveAvgPoolNd.

        This function handles GEOMETRY ONLY:
            - Computing receptive field ranges per output position
            - Extracting input patches
            - Accumulating results back into input space

        All attribution logic is delegated to _distribute_attribution.
        Changing the attribution rule there automatically propagates here.

        Attribution signal: |a_j * 1| = |a_j|  (unit weights, w=1)
        This reflects that avgpool has no learned weights — each input
        position contributes proportionally to its activation magnitude.

        Overlapping receptive fields (when in_size not divisible by out_size)
        are handled correctly via += accumulation — a pixel covered by
        multiple output neurons accumulates mass from all of them,
        consistent with PAF's general += rule.

        Args:
            a_in:             input activations, shape (N, C, *in_spatial)
            dist_out:         output distribution, shape (N, C, *out_spatial)
            dist_out_signed:  signed output distribution, same shape as dist_out,
                              or None to skip signed computation entirely
            spatial_start_dim: index of first spatial dimension (2 for conv features)

        Returns:
            dist_in:        unsigned input distribution, shape (N, C, *in_spatial)
            dist_in_signed: signed input distribution, same shape, or None
        """
        # --- Shape extraction ---
        in_spatial  = list(a_in.shape[spatial_start_dim:])    # e.g. [7, 7]
        out_spatial = list(dist_out.shape[spatial_start_dim:]) # e.g. [1, 1]
        n_spatial   = len(in_spatial)                          # e.g. 2

        # --- Validation ---
        assert a_in.shape[:spatial_start_dim] == dist_out.shape[:spatial_start_dim], (
            f"Batch/channel dims mismatch: "
            f"a_in={a_in.shape}, dist_out={dist_out.shape}"
        )
        assert dist_out.dim() == spatial_start_dim + n_spatial, (
            f"dist_out rank {dist_out.dim()} inconsistent with "
            f"spatial_start_dim={spatial_start_dim}, n_spatial={n_spatial}. "
            f"Expected rank {spatial_start_dim + n_spatial}."
        )

        # --- Precompute receptive field ranges — geometry only ---
        # ranges_per_dim[d][i] = (start, end) for output position i along dim d
        # Uses PyTorch's exact adaptive avg pool formula
        ranges_per_dim = [
            self.get_adaptive_avg_pool_ranges(in_spatial[d], out_spatial[d])
            for d in range(n_spatial)
        ]

        # --- Output accumulators ---
        dist_in        = torch.zeros_like(a_in)

        # --- Iterate over all output spatial positions ---
        # product(*[range(s) for s in out_spatial]) generates all output positions
        # Example: out_spatial=[3,3] → (0,0),(0,1),(0,2),(1,0),...,(2,2)
        # Example: out_spatial=[1,1] → (0,0) only — single iteration for global pool
        for out_pos in product(*[range(s) for s in out_spatial]):

            # --- Step 1: Build index into dist_out at this output position ---
            # (slice(None),) * spatial_start_dim → (:, :) selects all N and C
            # tuple(out_pos) → (i, j) selects specific spatial position
            # Combined: dist_out[:, :, i, j] → shape (N, C)
            mass_idx = (slice(None),) * spatial_start_dim + tuple(out_pos)
            # mass_idx example: (:, :, 1, 2) for out_pos=(1,2) with spatial_start_dim=2

            # --- Step 2: Build receptive field slices in input space ---
            # ranges_per_dim[d][out_pos[d]] gives (start, end) for dim d
            # slice(start, end) selects those input positions
            rf_slices = tuple(
                slice(
                    ranges_per_dim[d][out_pos[d]][0],  # start index
                    ranges_per_dim[d][out_pos[d]][1]   # end index
                )
                for d in range(n_spatial)
            )
            # rf_slices example: (slice(2,5), slice(4,7)) for out_pos=(1,2)

            # --- Step 3: Index into a_in to get the receptive field patch ---
            # act_idx combines batch/channel selectors with spatial rf_slices
            act_idx   = (slice(None),) * spatial_start_dim + rf_slices
            # act_idx example: (:, :, 2:5, 4:7)
            act_patch = a_in[act_idx]
            # act_patch.shape example: (1, 512, 3, 3) for global pool with 7→1
            #                          (1, 256, 3, 3) for 7→3 at position (1,2)

            # --- Step 4: Flatten to _distribute_attribution format ---
            # batch_dims: non-spatial dims, e.g. (N, C) = (1, 512)
            batch_dims = act_patch.shape[:spatial_start_dim]

            # NC: total number of independent (n,c) slices
            # e.g. N=1, C=512 → NC=512
            # math.prod is cleaner than torch.prod for plain integers
            NC = math.prod(batch_dims)

            # patch_size: number of spatial elements per (n,c) slice
            # act_patch.reshape(NC, -1) merges all spatial dims into one
            # .shape[-1] gives the flattened spatial size
            # e.g. act_patch (1,512,3,3) → reshape(512,-1) → (512,9) → patch_size=9
            patch_size = act_patch.reshape(NC, -1).shape[-1]

            # Reshape to engine format:
            # a_standard: (NC, 1, patch_size, 1)
            #   dim0=NC:         independent (n,c) slices processed in parallel
            #   dim1=1:          C_out=1 (avgpool has one "output channel" per input channel)
            #   dim2=patch_size: all input positions in this receptive field
            #   dim3=1:          L=1 (single output position per iteration)
            a_standard = act_patch.reshape(NC, 1, patch_size, 1)

            # w_standard: (1, 1, patch_size, 1)
            # Unit weights: avgpool has w=1 for all positions
            # Attribution signal: |a_j * 1| = |a_j|
            # patch_size determined per-iteration since rf sizes can vary
            # (non-uniform when in_size not divisible by out_size)
            w_standard = torch.ones(
                1, 1, patch_size, 1,
                device=a_in.device,
                dtype=a_in.dtype
            )

            # dist_standard: (NC, 1, 1, 1)
            # Mass at this output position for each (n,c) slice
            dist_standard = dist_out[mass_idx].reshape(NC, 1, 1, 1)

            # --- Step 5: Delegate to attribution engine ---
            # _distribute_attribution computes:
            #   resp[j] = |a_j| / sum_j |a_j|  (with unit weights w=1)
            #   p_patch[j] = resp[j] * dist_standard
            # If attribution formula changes in _distribute_attribution,
            # it automatically applies here — no changes needed in this function
            p_patch = self.engine.distribute_attribution(
                dist_out        = dist_standard,   # (NC, 1, 1, 1)
                a_patches       = a_standard,      # (NC, 1, patch_size, 1)
                w_flat          = w_standard,      # (1,  1, patch_size, 1)
                score_fn=score_fn
            )
            # p_patch:      (NC, patch_size, 1) — unsigned mass per input position
            # signed_patch: (NC, patch_size, 1) or None

            # --- Step 6: Reshape and accumulate into output tensor ---
            # rf_sizes: spatial shape of this receptive field
            # e.g. (3, 3) for a 3x3 patch
            rf_sizes = act_patch.shape[spatial_start_dim:]

            # p_patch (NC, patch_size, 1)
            # → squeeze(-1):           (NC, patch_size)
            # → reshape(*batch_dims, *rf_sizes): (N, C, *rf_sizes)
            #   e.g. (1, 512, 3, 3) for global pool
            p_reshaped = p_patch.squeeze(-1).reshape(*batch_dims, *rf_sizes)

            # += accumulates contributions from overlapping receptive fields
            # correctly implements PAF's general += rule
            dist_in[act_idx] += p_reshaped

        # --- Mass conservation check ---
        # Should hold exactly since _distribute_attribution conserves mass
        # and += accumulation is correct for overlapping fields
        mass_loss = (dist_out.sum() - dist_in.sum()).abs().item()
        if mass_loss > 1e-3:
            print(
                f"WARNING adaptive_avgpool_exact_inverse mass loss = {mass_loss:.6f} "
                f"(out={dist_out.sum():.4f}, in={dist_in.sum():.4f}) "
                f"in_spatial={in_spatial}, out_spatial={out_spatial}"
            )

        return dist_in

    def get_adaptive_avg_pool_ranges(self, in_size: int, out_size: int):
        """Compute (start, end) ranges for each output position."""
        return [(math.floor(i * in_size / out_size), 
                math.ceil((i + 1) * in_size / out_size)) 
                for i in range(out_size)]

    def paf_propagate_add(
        self,
        preds:      List[str],
        dist_out:   torch.Tensor,
        act_preds:  List[torch.Tensor],
        score_fn:   callable,
    ) -> Dict[str, torch.Tensor]:
        """
        PAF backward through additive merge.
        Returns {pred_name: distribution} — caller handles storage.
        All scoring delegated to _distribute_attribution and score_fn.
        """
        B         = len(preds)
        flat_size = dist_out.numel()

        if B == 1:
            return {preds[0]: dist_out}

        same_shape = all(a.shape == act_preds[0].shape for a in act_preds)

        if same_shape:
            # Element-wise routing — each output position has its own
            # branch activations. _distribute_attribution handles scoring.
            act_stacked = torch.stack(
                [a.reshape(flat_size) for a in act_preds], dim=1
            )                                      # (flat_size, B)

            a_standard = act_stacked.reshape(flat_size, 1, B, 1)
            w_standard = torch.ones(1, 1, B, 1,
                                    device=dist_out.device,
                                    dtype=dist_out.dtype)
            d_standard = dist_out.reshape(flat_size, 1, 1, 1)

            p_in = self.engine.distribute_attribution(
                dist_out  = d_standard,
                a_patches = a_standard,
                w_flat    = w_standard,
                score_fn  = score_fn,
            ).squeeze(-1)                          # (flat_size, B)

            return {
                pred: p_in[:, i].reshape(dist_out.shape)
                for i, pred in enumerate(preds)
            }

        else:
            # Different shapes — pad smaller branches with zeros
            # Zeros score zero via any score_fn: score(0, w) = 0
            # Padding does not distort — it only fills missing positions
            # _distribute_attribution sees all branches and scores them
            # No manual mass pre-distribution
            act_flat = [a.reshape(-1) for a in act_preds]
            max_size = max(a.numel() for a in act_flat)

            act_padded = torch.stack([
                torch.nn.functional.pad(a, (0, max_size - a.numel()))
                for a in act_flat
            ], dim=0)                              # (B, max_size)

            # Layout: treat all branch elements as patch positions
            # patch_size = B * max_size would mix branches — wrong
            #
            # Correct layout:
            # For each output position (flat_size=1 since we treat globally):
            #   patch_size = B (one position per branch)
            #   each branch represented by its L2-style score from score_fn
            #
            # Pass raw padded activations per branch as separate columns
            # a_patches: (1, 1, B, max_size) — B branches, max_size elements
            # score_fn:  scores each (branch, element) pair
            # K = scores.abs().sum(dim=2) — sums over B branches ← wrong
            #
            # The issue: _distribute_attribution sums over dim=2 (patch_size=B)
            # but we want to sum over max_size (elements within branch)
            # then use the per-branch sum as the score
            #
            # Solution: transpose so patch_size=max_size, L=B
            # a_patches: (1, 1, max_size, B)
            # score_fn sees max_size elements for each of B output channels
            # K[b] = sum_j score(a_bj, w_j) — per-branch sum ✓
            # resp[b,j] = score(a_bj, w_j) / K[b]
            # p_in[j] = sum_b resp[b,j] * d_out[b]
            #
            # d_standard: (1, B, 1, 1) — one mass per branch
            # But we want to distribute dist_out (flat_size elements)
            # across B branches — so d_standard = dist_out total

            a_standard = act_padded.unsqueeze(0).unsqueeze(0)  # (1,1,B,max_size)
            w_standard = torch.ones(1, 1, B, max_size,
                                    device=dist_out.device,
                                    dtype=dist_out.dtype)

            # dist_out as single scalar — total mass to distribute
            d_standard = dist_out.abs().sum().reshape(1, 1, 1, 1)

            # _distribute_attribution with patch_size=B, L=max_size:
            # scores: (1, 1, B, max_size)
            # K = scores.abs().sum(dim=2) = sum over B ← still wrong dim
            #
            # The fundamental problem: _distribute_attribution always
            # sums over dim=2 (patch_size). For different-shape branches
            # we need to sum over elements within each branch (max_size dim)
            # then use per-branch total as the score.
            #
            # Only correct layout:
            # patch_size = B (one score per branch)
            # Each branch score = score_fn applied to entire branch flattened
            # Reduce each branch to one value USING score_fn semantics:
            # branch_score_b = score_fn(act_b, w=1).sum()

            branch_scores = []
            for act_b in act_padded:
                a_b = act_b.reshape(1, 1, max_size, 1)
                w_b = torch.ones(1, 1, max_size, 1,
                                device=dist_out.device,
                                dtype=dist_out.dtype)
                scores, _ = score_fn(a_b, w_b)    # (1, 1, max_size, 1)
                # For pre_normalised score_fns, scores sum to 1 — use sum of raw (unnormalised) product instead
                if getattr(score_fn, 'pre_normalised', False):
                    # Re-compute without normalisation for branch scoring
                    raw = a_b.abs() * w_b.abs()
                    branch_scores.append(raw.sum())
                else:
                    branch_scores.append(scores.sum())
                #branch_scores.append(scores.sum()) # scalar — total branch score

            # Stack branch scores: (B,) — one score per branch
            branch_scores = torch.stack(branch_scores)  # (B,)

            # Now use _distribute_attribution with patch_size=B
            # a_patches: (1, 1, B, 1) — one pre-computed score per branch
            # score_fn will be called again inside — but with unit activations
            # since we already computed scores above
            #
            # This double-applies score_fn which is wrong.
            # The correct approach: normalise branch_scores directly
            # using the same is_zero logic as _distribute_attribution

            K       = branch_scores.abs().sum()
            is_zero = K <= self.eps
            if is_zero:
                resp = torch.ones(B, device=dist_out.device,
                                dtype=dist_out.dtype) / B
            else:
                resp = branch_scores / K           # (B,) sums to 1

            # Distribute dist_out by resp
            total = dist_out.abs().sum()
            return {
                pred: (total * resp[i]).reshape(act.shape) \
                    if act.numel() == 1 \
                    else (dist_out * resp[i]).reshape(act.shape)
                for i, (pred, act) in enumerate(zip(preds, act_preds))
            }

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
        pure_param = self.graph_info.pure_parameter_nodes

        tensor_branches = [(n, a) for n, a in pred_pairs if a is not None]
        scalar_branches = [(n, a) for n, a in pred_pairs if a is None]
        image_branches  = [(n, a) for n, a in tensor_branches
                        if n not in pure_param]
        param_branches  = [(n, a) for n, a in tensor_branches
                        if n in pure_param]

        # Zero for scalar and parameter branches
        dist_zero_map={}
        for name, _ in scalar_branches + param_branches:
            dist_zero_map[name]=torch.zeros_like(dist_out)
            #self._store_edge(name, curr_layer,
            #                torch.zeros_like(dist_out),
            #                distributions, False, mode_key)

        if not image_branches:
            return dist_zero_map

        if self.redistribute_param_mass or not param_branches:
            # Pixel attribution OR no param branches (ResNet):
            # Route only among image branches
            preds    = [n for n, _ in image_branches]
            act_list = [a for _, a in image_branches]

            dist_map = self.paf_propagate_add(
                preds     = preds,
                dist_out  = dist_out,
                act_preds = act_list,
                score_fn  = score_fn,
            )
        else:
            # Model analysis mode — include param branches as activation
            # Treat param activation as weight in score_fn
            # Only valid for single image + single param, same shape
            img_name, a_img = image_branches[0]
            prm_name, a_prm = param_branches[0]

            preds    = [img_name, prm_name]
            act_list = [a_img, a_prm]

            dist_map = self.paf_propagate_add(
                preds     = preds,
                dist_out  = dist_out,
                act_preds = act_list,
                score_fn  = score_fn,
            )
        return dist_map | dist_zero_map


    def paf_propagate_mhsa(
        self,
        dist_out:  torch.Tensor,
        a_in:      torch.Tensor,
        layer:     nn.MultiheadAttention,
        score_fn:  callable,
    ) -> torch.Tensor:
        with torch.no_grad():
            if a_in.dim() == 2:
                a_in     = a_in.unsqueeze(0)
            if dist_out.dim() == 2:
                dist_out = dist_out.unsqueeze(0)

            B, S, D = a_in.shape
            n_heads  = layer.num_heads
            d_head   = D // n_heads

            # Recompute Q, K, V
            W_in = layer.in_proj_weight
            b_in = layer.in_proj_bias
            W_Q, W_K, W_V = W_in[:D], W_in[D:2*D], W_in[2*D:]
            b_Q = b_in[:D]   if b_in is not None else 0
            b_K = b_in[D:2*D] if b_in is not None else 0
            b_V = b_in[2*D:]  if b_in is not None else 0

            Q = a_in @ W_Q.T + b_Q   # (B, S, D)
            K = a_in @ W_K.T + b_K
            V = a_in @ W_V.T + b_V

            def to_heads(t):
                return t.view(B, S, n_heads, d_head).transpose(1, 2)

            A = torch.softmax(
                to_heads(Q) @ to_heads(K).transpose(-2, -1) / (d_head ** 0.5),
                dim=-1
            )                          # (B, H, S_out, S_in)
            A_avg = A.mean(dim=1)      # (B, S_out, S_in)

            # ----------------------------------------------------------------
            # Step 1: Route through attention via _distribute_attribution
            #
            # For each output token i: distribute dist_out[i] to input
            # tokens j proportional to A_ij * |V_j|
            #
            # Format: treat S_in as patch positions, one output per token
            #   a_patches: (B*S_out, 1,    S_in, 1) — routing signal per token
            #   w_flat:    (1,       1,    S_in, 1) — unit weights
            #   dist_out:  (B*S_out, 1,    1,    1) — mass per output token
            # ----------------------------------------------------------------
            V_mag    = V.abs().mean(dim=-1)                    # (B, S_in)
            routing  = A_avg * V_mag.unsqueeze(1)              # (B, S_out, S_in)

            BS_out   = B * S
            a_att    = routing.reshape(BS_out, 1, S, 1)        # (BS, 1, S_in, 1)
            w_att    = torch.ones(1, 1, S, 1,
                                device=a_in.device,
                                dtype=a_in.dtype)
            # Token mass: sum dist_out over D for each token
            d_att    = dist_out.abs().sum(dim=-1)\
                                .reshape(BS_out, 1, 1, 1)      # (BS, 1, 1, 1)

            # p_att: (BS_out, S_in, 1) — mass per input token per output token
            p_att    = self.engine.distribute_attribution(
                dist_out  = d_att,
                a_patches = a_att,
                w_flat    = w_att,
                score_fn  = score_fn,
            )

            # Sum over output tokens to get total mass per input token
            # p_att: (BS_out, S_in, 1) → reshape (B, S_out, S_in) → sum S_out
            token_dist = p_att.squeeze(-1)\
                            .reshape(B, S, S)\
                            .sum(dim=1)                      # (B, S_in)

            # ----------------------------------------------------------------
            # Step 2: Route through V projection via _distribute_attribution
            #
            # V_j = a_j @ W_V  (D input features → D output features)
            # For each input token j, distribute token_dist[j] across
            # its D input features proportional to score_fn(a_j, W_V_j)
            #
            # Format: one token at a time, D features as patch positions
            #   a_patches: (B*S, 1, D, 1) — input features per token
            #   w_flat:    (1,   1, D, 1) — unit weights (not W_V!)
            #   dist_out:  (B*S, 1, 1, 1) — token mass (scalar per token)
            #
            # Note: we use unit weights here because W_V routing was
            # already captured in Step 1 via V_mag = |a @ W_V|.mean()
            # Using W_V again would double-count weight information.
            # ----------------------------------------------------------------
            a_flat   = a_in.reshape(B * S, D)                 # (B*S, D)
            a_v      = a_flat.reshape(B * S, 1, D, 1)         # (B*S, 1, D, 1)
            w_v      = torch.ones(1, 1, D, 1,
                                device=a_in.device,
                                dtype=a_in.dtype)
            d_v      = token_dist.reshape(B * S, 1, 1, 1)     # (B*S, 1, 1, 1)

            # p_v: (B*S, D, 1) — mass per input feature per token
            p_v      = self.engine.distribute_attribution(
                dist_out  = d_v,
                a_patches = a_v,
                w_flat    = w_v,
                score_fn  = score_fn,
            )

            dist_in  = p_v.squeeze(-1).reshape(B, S, D)       # (B, S, D)

            if self.debug_level:
                loss = abs(dist_in.sum().item() - dist_out.sum().item())
                if loss > 1e-4:
                    print(f"WARNING MHSA mass loss={loss:.6f} "
                        f"out={dist_out.sum():.4f} "
                        f"in={dist_in.sum():.4f}")

            return dist_in.squeeze(0)

    def paf_propagate_cls_token(
        self,
        dist_out: torch.Tensor,    # (D,) or (1, D) — distribution after x[:,0]
        a_in:     torch.Tensor,    # (B, S, D) — full sequence before slicing
    ) -> torch.Tensor:
        """
        CLS token extraction: x[:, 0] selects only token 0.
        Routes all mass back to token position 0.
        All other token positions receive zero mass.
        """
        if dist_out.dim() == 1:
            dist_out = dist_out.unsqueeze(0)    # (1, D)

        B, S, D = a_in.shape if a_in.dim() == 3 \
                else (1, a_in.shape[0], a_in.shape[1])

        dist_in         = torch.zeros(B, S, D,
                                    device=dist_out.device,
                                    dtype=dist_out.dtype)
        dist_in[:, 0, :] = dist_out            # all mass to CLS position
        return dist_in.squeeze(0)              # (S, D)


    def paf_propagate_reshape_permute(
        self,
        dist_out: torch.Tensor,
        target_shape: tuple,
    ) -> torch.Tensor:
        """
        Passthrough for reshape/permute — just reshape distribution
        to match input activation shape.
        Mass is conserved by construction.
        """
        return dist_out.reshape(target_shape)

    def paf_propagate_cat_node(
        self,
        curr_layer:    str,
        dist_out:      torch.Tensor,
        pred_pairs:    List[Tuple[str, Optional[torch.Tensor]]],
        distributions: Dict[str, torch.Tensor],
        has_skip:      bool,
        mode_key:      tuple,
        score_fn:      callable,
        store_edge_fn: callable,
    ) -> None:
        """
        PAF backward through torch.cat node.

        Two modes controlled by self.redistribute_param_mass:

        True  (default — pixel attribution):
            Parameter branches (e.g. CLS token) receive zero.
            Their mass is recovered by scaling image branch distribution
            so total mass is conserved. _distribute_attribution routes
            scaled mass through image branch using score_fn.

        False (model analysis):
            Exact slice per branch — no redistribution.
            class_token retains its mass fraction, showing how much
            of the prediction comes from learned prior vs input pixels.
        """
        tensor_branches = [(n, a) for n, a in pred_pairs if a is not None]
        scalar_branches = [(n, a) for n, a in pred_pairs if a is None]

        # Get cat argument order and dim from FX node
        fx_node = self.graph_info.node_map.get(curr_layer)

        # fx_node is always valid — curr_layer comes from backward_order
        # which is built from the same FX graph
        cat_inputs = [
            n.name for n in fx_node.args[0]
            if isinstance(n, torch.fx.Node)
        ]
        cat_dim = fx_node.args[1] if len(fx_node.args) > 1 else 1

        # cat_dim passed as node (dynamic dim) — detect from shapes
        if isinstance(cat_dim, torch.fx.Node):
            cat_dim = self._detect_cat_dim(dist_out, tensor_branches)

        # O(1) activation lookup
        act_map = {name: act for name, act in tensor_branches}

        # Classify branches — parameter or image-derived
        pure_param_nodes = self.graph_info.pure_parameter_nodes
        param_branch_names = {
            name for name, _ in tensor_branches
            if name in pure_param_nodes
        }
        image_branch_names = {
            name for name, _ in tensor_branches
            if name not in param_branch_names
        }

        # Slice dist_out in correct FX cat order
        slices_map = {}
        idx = 0
        for name in cat_inputs:
            act = act_map.get(name)
            if act is None:
                continue   # scalar input — skip
            size   = act.shape[cat_dim]
            s      = [slice(None)] * dist_out.dim()
            s[cat_dim] = slice(idx, idx + size)
            slices_map[name] = dist_out[tuple(s)]   # view — no copy
            idx += size

        # ----------------------------------------------------------------
        # Mode 1: redistribute_param_mass=True — pixel attribution
        # ----------------------------------------------------------------
        if self.redistribute_param_mass:
            total_mass = dist_out.sum()

            # Sum of image branch slices only
            image_sum = sum(
                slices_map[name].sum()
                for name in image_branch_names
                if name in slices_map
            )

            # Scale factor — image branches absorb full output mass
            # If image_sum=0 (degenerate case), no scaling possible
            scale = (total_mass / image_sum) \
                    if image_sum.abs() > self.eps \
                    else torch.ones(1, device=dist_out.device,
                                    dtype=dist_out.dtype)

            for name in cat_inputs:
                if name not in slices_map:
                    continue

                act = act_map.get(name)
                if act is None:
                    continue

                if name in param_branch_names:
                    # Parameter branch — zero
                    # Mass already recovered via scale on image branches
                    dist = torch.zeros_like(slices_map[name])
                    store_edge_fn(
                        prev_layer    = name,
                        curr_layer    = curr_layer,
                        dist          = dist,
                        distributions = distributions,
                        skip          = False,   # no accumulation for param
                        mode_key      = mode_key,
                    )

                else:
                    dist_scaled = slices_map[name] * scale    # (1, 196, 768)
                    act         = act_map[name]               # (1, 196, 768)
                    patch_size  = act.numel()

                    # Single output position distributes to all patch elements
                    # d_standard: scalar total mass (1, 1, 1, 1)
                    # a_standard: all patch activations as patch positions
                    a_standard = act.reshape(1, 1, patch_size, 1)
                    w_standard = torch.ones(1, 1, patch_size, 1,
                                            device=dist_out.device,
                                            dtype=dist_out.dtype)
                    d_standard = dist_scaled.abs().sum().reshape(1, 1, 1, 1)

                    p_in = self.engine.distribute_attribution(
                        dist_out  = d_standard,   # (1, 1, 1,          1)
                        a_patches = a_standard,   # (1, 1, patch_size,  1)
                        w_flat    = w_standard,   # (1, 1, patch_size,  1)
                        score_fn  = score_fn,
                    )                             # (1, patch_size, 1)

                    dist = p_in.squeeze(-1).reshape(act.shape)

                    store_edge_fn(
                        prev_layer    = name,
                        curr_layer    = curr_layer,
                        dist          = dist,
                        distributions = distributions,
                        skip          = has_skip,
                        mode_key      = mode_key,
                    )

        # ----------------------------------------------------------------
        # Mode 2: redistribute_param_mass=False — model analysis
        # Exact slice per branch — class_token retains its mass fraction
        # Shows: how much prediction comes from learned prior vs input
        # ----------------------------------------------------------------
        else:
            for name in cat_inputs:
                if name not in slices_map:
                    continue
                store_edge_fn(
                    prev_layer    = name,
                    curr_layer    = curr_layer,
                    dist          = slices_map[name],
                    distributions = distributions,
                    skip          = has_skip,
                    mode_key      = mode_key,
                )

        # Zero for scalar predecessors (shape args etc.)
        for name, _ in scalar_branches:
            store_edge_fn(
                prev_layer    = name,
                curr_layer    = curr_layer,
                dist          = torch.zeros_like(dist_out),
                distributions = distributions,
                skip          = False,
                mode_key      = mode_key,
            )

        if self.debug_level:
            active_names = image_branch_names \
                        if self.redistribute_param_mass \
                        else (image_branch_names | param_branch_names)
            total = sum(
                self.edge_mass[mode_key].get((n, curr_layer), 0)
                for n in active_names
            )
            loss = abs(total - dist_out.sum().item())
            if loss > 1e-4:
                print(
                    f"WARNING CAT mass loss={loss:.6f} at {curr_layer} "
                    f"(mode=redistribute={self.redistribute_param_mass})"
                )

    def _detect_cat_dim(
        self,
        dist_out:     torch.Tensor,
        tensor_pairs: List[Tuple[str, torch.Tensor]],
    ) -> int:
        """Fallback cat_dim detection when dim is not a literal in FX args."""
        for dim in range(dist_out.dim()):
            try:
                sizes = [a.shape[dim] for _, a in tensor_pairs if a.dim() > dim]
                if sum(sizes) == dist_out.shape[dim]:
                    return dim
            except IndexError:
                continue
        return 1   # ViT default