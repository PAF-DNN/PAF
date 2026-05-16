"""
Attribution Engine Module
==========================
Core attribution computation engine for PAF (Probabilistic Activation Flow).
Provides memory-efficient chunked attribution propagation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

# Public alias so LayerPropagator can call engine.distribute_attribution

class AttributionEngine:
    """
    Attribution computation engine with automatic memory-efficient chunking.

    Handles the core attribution distribution logic for PAF, automatically
    chunking computations over output channels and sequence length to stay
    within specified memory budgets.
    """

    def __init__(self, eps: float = 1e-9):
        """
        Initialize the attribution engine.

        Parameters
        ----------
        eps : float
            Stabilisation constant for lifting. Also acts as interpretive dial:
            small eps -> suppressed neurons invisible,
            large eps -> suppressed neurons contribute.
        """
        assert eps > 0, "eps must be positive"
        self.eps = eps

    def distribute_attribution(
        self,
        dist_out:   torch.Tensor,          # (N, C_out, 1,          L)
        a_patches:  torch.Tensor,          # (N, 1,     patch_size,  L)
        w_flat:     torch.Tensor,          # (1, C_out, patch_size,  1)
        score_fn:   Callable,
        mask_exp:   Optional[torch.Tensor] = None,
        memory_budget_mb: int = 1000,       # max MB per chunk
    ) -> torch.Tensor:
        """
        PAF attribution engine — automatically chunks over C_out and L
        to stay within memory budget.

        For small layers: processes in one pass (no overhead).
        For large layers: chunks to avoid OOM.
        """
        N, C_out, _, L = dist_out.shape
        patch_size = a_patches.shape[2]
        bytes_per_el = 4   # float32

        # Compute chunk sizes to stay within budget
        budget_bytes = memory_budget_mb * 1024 * 1024

        # Chunk C_out first — reduces peak tensor size
        chunk_Cout = max(1, budget_bytes // (
            N * patch_size * L * bytes_per_el
        ))
        chunk_Cout = min(chunk_Cout, C_out)

        # Then chunk L if still too large
        chunk_L = max(1, budget_bytes // (
            N * chunk_Cout * patch_size * bytes_per_el
        ))
        chunk_L = min(chunk_L, L)

        # Single pass — no chunking needed
        if chunk_Cout >= C_out and chunk_L >= L:
            return self._run_attribution(
                dist_out, a_patches, w_flat, score_fn, mask_exp
            )

        # Chunked pass
        dist_in = torch.zeros(
            N, patch_size, L,
            device=a_patches.device,
            dtype=a_patches.dtype,
        )

        with torch.no_grad():
            for c_start in range(0, C_out, chunk_Cout):
                c_end   = min(c_start + chunk_Cout, C_out)
                d_chunk = dist_out[:, c_start:c_end, :, :]
                w_chunk = w_flat[:,   c_start:c_end, :, :]

                for l_start in range(0, L, chunk_L):
                    l_end    = min(l_start + chunk_L, L)
                    a_chunk  = a_patches[..., l_start:l_end]
                    dl_chunk = d_chunk[...,   l_start:l_end]
                    m_chunk  = mask_exp[...,  l_start:l_end] \
                            if mask_exp is not None else None

                    result = self._run_attribution(
                        dl_chunk, a_chunk, w_chunk, score_fn, m_chunk
                    )
                    dist_in[..., l_start:l_end].add_(result)

                    del result, a_chunk, dl_chunk
                    if m_chunk is not None:
                        del m_chunk

                del d_chunk, w_chunk

            if a_patches.device.type == 'cuda':
                torch.cuda.empty_cache()

        return dist_in

    def _run_attribution(
        self,
        dist_out:  torch.Tensor,   # (N, C_out, 1,          L)
        a_patches: torch.Tensor,   # (N, 1,     patch_size,  L)
        w_flat:    torch.Tensor,   # (1, C_out, patch_size,  1)
        score_fn:  Callable,
        mask_exp:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Core attribution logic — no chunking.
        Called by distribute_attribution for each chunk.
        """
        '''
        import traceback
        pre = getattr(score_fn, 'pre_normalised', False)
        if pre:
            print(f"pre_normalised=True CONFIRMED in _run_attribution")
            traceback.print_stack(limit=4)  # show call chain
        '''
        with torch.no_grad():
            #print(f"[ENGINE] score_fn={score_fn.__name__ if hasattr(score_fn, '__name__') else score_fn}, "
            #  f"pre_normalised={getattr(score_fn, 'pre_normalised', False)}")
            scores, _ = score_fn(a_patches, w_flat)

            if mask_exp is not None:
                scores.mul_(mask_exp)

            if not getattr(score_fn, 'pre_normalised', False):
                K = scores.abs().sum(dim=2, keepdim=True)
                is_safe = K > self.eps
                K.masked_fill_(~is_safe, 1.0)
                scores.div_(K)
                scores.masked_fill_(~is_safe, 0.0)
            
            scores.mul_(dist_out)

            return scores.sum(dim=1)   # (N, patch_size, L)

    def _get_chunk_sizes(
        self,
        N: int, C_out: int, patch_size: int, L: int
    ) -> Tuple[int, int]:
        """
        Automatically select chunk sizes to stay under memory budget.
        Target: peak tensor < 500 MB.
        """
        budget_bytes  = 500 * 1024 * 1024   # 500 MB
        bytes_per_el  = 4                    # float32

        # Full tensor size
        full_bytes = N * C_out * patch_size * L * bytes_per_el

        if full_bytes <= budget_bytes:
            return C_out, L   # no chunking needed

        # Chunk L first (cheaper — no accumulation needed)
        chunk_L    = max(1, budget_bytes // (N * C_out * patch_size * bytes_per_el))
        chunk_L    = min(chunk_L, L)

        # If still too large, also chunk C_out
        remaining  = budget_bytes // (N * patch_size * chunk_L * bytes_per_el)
        chunk_Cout = max(1, min(remaining, C_out))

        return chunk_Cout, chunk_L


