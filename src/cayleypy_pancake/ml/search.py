from __future__ import annotations

from heapq import nsmallest
from typing import Callable, Iterable, List, Optional
import numpy as np

from cayleypy_pancake.search import apply_move_copy, gap_h, is_solved


def beam_improve_with_ml(
    perm: Iterable[int],
    h_fn: Callable[[List[List[int]]], np.ndarray],
    *,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    beam_width: int = 256,
    depth: int = 192,
    w: float = 0.5,
    w_gap: float = 0.15,
    gap_mode: str = "log1p",
    patience: Optional[int] = None,
) -> List[int]:
    start = list(perm)
    base_moves = baseline_moves_fn(start)
    best_len = len(base_moves)
    if best_len == 0:
        return []

    n = len(start)
    k_values = range(2, n + 1)

    beam = [(0.0, 0, start, [])]  # (f, g, state, path)
    best_path = None
    best_g = {tuple(start): 0}
    no_improve_steps = 0

    for _step in range(depth):
        cand_states: List[List[int]] = []
        cand_meta = []

        for _f, g, state, path in beam:
            if g >= best_len:
                continue
            for k in k_values:
                new_g = g + 1
                if new_g >= best_len:
                    continue

                nxt = apply_move_copy(state, k)
                key = tuple(nxt)

                prev = best_g.get(key)
                if prev is not None and prev <= new_g:
                    continue
                best_g[key] = new_g

                new_path = path + [k]

                if is_solved(nxt):
                    best_len = new_g
                    best_path = new_path
                    no_improve_steps = 0
                    continue

                cand_states.append(nxt)
                cand_meta.append((new_g, nxt, new_path))

        if not cand_states:
            break

        h_ml = h_fn(cand_states).astype(np.float32, copy=False)

        gvals = np.fromiter((gap_h(s) for s in cand_states), dtype=np.float32, count=len(cand_states))
        if gap_mode == "log1p":
            gvals = np.log1p(gvals)
        elif gap_mode == "norm":
            gvals = gvals / max(1.0, float(n))
        elif gap_mode == "none":
            pass
        else:
            raise ValueError("gap_mode must be: none | log1p | norm")

        h = w * h_ml + w_gap * gvals

        candidates = [(new_g + float(hh), new_g, nxt, new_path)
                      for (new_g, nxt, new_path), hh in zip(cand_meta, h)]
        beam = nsmallest(beam_width, candidates, key=lambda x: x[0])

        no_improve_steps += 1
        if patience is not None and no_improve_steps >= patience:
            break

    return best_path if best_path is not None else base_moves
