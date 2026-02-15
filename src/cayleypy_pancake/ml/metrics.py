from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def sanity_corr_bfs(model, graph, rw_length: int, width: int = 2000, nbt_history_depth: int = 1) -> float:
    model.eval()
    X_s, y_s = graph.random_walks(width=width, length=rw_length, mode="bfs", nbt_history_depth=nbt_history_depth)
    pred = model(X_s.long()).float().view(-1).detach().cpu().numpy()
    y_raw = y_s.float().view(-1).detach().cpu().numpy()
    y_t = np.log1p(y_raw)
    return float(np.corrcoef(pred, y_t)[0, 1])
