from __future__ import annotations

import numpy as np
import torch


def y_transform_torch(y: torch.Tensor, mode: str | None) -> torch.Tensor:
    y = y.detach().float().view(-1)
    if mode is None or mode == "none":
        return y
    if mode == "log1p":
        return torch.log1p(y)
    if mode == "norm_max":
        return y / y.max().clamp_min(1.0)
    raise ValueError(f"Unknown y_transform={mode}")


def make_bins_np(y: torch.Tensor, clip: int = 60, bin_size: float = 1.0) -> np.ndarray:
    y_np = y.detach().float().cpu().numpy()
    b = np.floor(y_np / float(bin_size)).astype(np.int32)
    b = np.clip(b, 0, int(clip))
    return b


def gen_walks(cfg: dict, graph, mode: str):
    nbt_hist = int(cfg.get("nbt_history_depth", cfg.get("history_depth", 1)))
    return graph.random_walks(
        width=int(cfg["rw_width"]),
        length=int(cfg["rw_length"]),
        mode=mode,
        nbt_history_depth=nbt_hist,
    )
