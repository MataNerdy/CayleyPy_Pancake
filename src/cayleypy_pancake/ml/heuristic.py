from __future__ import annotations

from typing import Sequence
import numpy as np
import torch


class MLHeuristic:
    """Returns float score per state. Smaller = better."""
    def __init__(self, model: torch.nn.Module, device: str | torch.device, batch_size: int = 8192):
        self.model = model
        self.device = device
        self.bs = int(batch_size)

    @torch.no_grad()
    def __call__(self, states: Sequence[Sequence[int]]) -> np.ndarray:
        self.model.eval()
        out = np.empty(len(states), dtype=np.float32)
        i = 0
        while i < len(states):
            j = min(i + self.bs, len(states))
            x = torch.as_tensor(states[i:j], device=self.device, dtype=torch.long)
            y = self.model(x).float().view(-1)
            out[i:j] = y.detach().cpu().numpy().astype(np.float32)
            i = j
        return out
