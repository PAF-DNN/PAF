"""
PAF Runner
==========
Single source of truth for constructing PAF and extracting heatmaps.
All evaluation scripts use this — no direct PAF construction elsewhere.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.paf.paf import PAF
from core.paf.scoring import ScoringMode, make_mode_key
from core.paf.utils import make_mode_name
from Evaluation.eval_core.heatmap_utils import _paf_to_numpy_hw


class PAFRunner:
    """
    Constructs PAF and extracts heatmaps for evaluation.
    One instance per evaluation run — reused across samples.

    Eliminates repeated PAF construction boilerplate across
    randomization_test, pointing_game, and perturbation_test.
    """

    def __init__(
        self,
        model,
        graph_manager,
        paf_modes:   List[Tuple[ScoringMode, dict]],
        output_mode: str  = 'target',
        sample_idx:  int  = 0,
    ):
        self.model        = model
        self.graph_manager = graph_manager
        self.paf_modes    = paf_modes
        self.output_mode  = output_mode
        self.sample_idx   = sample_idx
        self._paf         = None  
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        if self._paf is not None:
            self._paf.cleanup()
            del self._paf
            self._paf = None
        torch.cuda.empty_cache()

    def run(
        self,
        x:            torch.Tensor,
        target_class: int,
        true_class:   Optional[int] = None,
    ) -> Dict[tuple, Dict]:
        """
        Run PAF on one sample. Returns distributions dict.
        Caller owns cleanup.
        """
        paf = PAF(
            model        = self.model,
            graph_manager = self.graph_manager,
            modes        = self.paf_modes,
            x            = x,
            target_class = target_class,
            true_class   = true_class or target_class,
            output_mode  = self.output_mode,
        )
        distributions = paf.distributions
        '''
        print("Stored keys:", list(paf.distributions.keys()))
        for mode, kwargs in self.paf_modes:
            t   = kwargs.get('tau', 1.0)
            key = (mode, t)
            print(f"Looking up: {key}, found: {key in paf.distributions}")
        '''
        return distributions

    def get_heatmaps(
        self,
        x:            torch.Tensor,
        target_class: int,
        eval_layer:   str,
        H:            int,
        W:            int,
        true_class:   Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run PAF and return {mode_name: heatmap_np} for all modes.
        Heatmap is (H, W) numpy array, sum=1.
        """
        distributions = self.run(x, target_class, true_class)
        heatmaps = {}

        for mode_key, store in distributions.items():
            mode_name = _mode_key_to_str(mode_key)
            p         = store.get(eval_layer)
            if p is None:
                heatmaps[mode_name] = np.zeros((H, W))
                continue
            '''
            print(f"[get_heatmaps] {mode_name}: "
                f"p.sum={p.sum().item():.6f}, "
                f"p.shape={p.shape}, "
                f"p.data_ptr={p.data_ptr()}") 
            '''
            heatmaps[mode_name] = _paf_to_numpy_hw(
                p[0] if p.dim() == 4 else p, H, W
            )
            '''
            print(f"[get_heatmaps] {mode_name}: "
                f"heatmap.sum={heatmaps[mode_name].sum():.6f}, "
                f"top_pixel={np.argmax(heatmaps[mode_name].flatten())}")
            '''
        return heatmaps

    @staticmethod
    def mode_names(paf_modes: List[Tuple]) -> List[str]:
        """Human-readable names for all modes."""
        return [_mode_key_to_str(make_mode_key(m, **kw))
                for m, kw in paf_modes]


def _mode_key_to_str(mode_key: tuple) -> str:
    parts = []
    for part in mode_key:
        if hasattr(part, 'value'):
            parts.append(str(part.value))
        elif isinstance(part, float):
            parts.append(f"t{part:.1f}")
        else:
            parts.append(str(part))
    return '_'.join(parts)
