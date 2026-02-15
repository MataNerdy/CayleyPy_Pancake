from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple, Dict, Any
import time
import numpy as np
import pandas as pd

from cayleypy_pancake.baseline import parse_permutation
from cayleypy_pancake.ml.search import beam_improve_with_ml


def build_rows_for_n(test_df: pd.DataFrame, target_n: int, k: int = 50, seed: int = 42) -> List[Tuple[int, List[int]]]:
    if "n" not in test_df.columns:
        tmp = test_df.copy()
        tmp["n"] = tmp["permutation"].apply(lambda x: len(parse_permutation(x)))
        sub = tmp[tmp["n"] == target_n].reset_index(drop=True)
    else:
        sub = test_df[test_df["n"] == target_n].reset_index(drop=True)

    if len(sub) == 0:
        raise ValueError(f"No rows with n={target_n}")

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sub), size=min(k, len(sub)), replace=False)

    rows: List[Tuple[int, List[int]]] = []
    for i in idx:
        rid = int(sub.loc[i, "id"])
        perm = parse_permutation(sub.loc[i, "permutation"])
        rows.append((rid, perm))
    return rows


def eval_ml_on_rows(
    rows: List[Tuple[int, List[int]]],
    *,
    h_ml,  # callable(states)->np.ndarray
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    beam_width: int = 256,
    depth: int = 192,
    w: float = 0.5,
    w_gap: float = 0.15,
    gap_mode: str = "log1p",
    patience: Optional[int] = None,
    log_every: int = 10,
) -> Dict[str, Any]:
    t0 = time.time()
    sum_base = 0
    sum_ml = 0
    improved = same = worse = 0
    total_gain_pos = 0
    max_gain = 0
    max_gain_id = None
    N = len(rows)

    for i, (rid, perm) in enumerate(rows, start=1):
        base = baseline_moves_fn(perm)
        lb = len(base)

        ml_moves = beam_improve_with_ml(
            perm, h_fn=h_ml,
            baseline_moves_fn=baseline_moves_fn,
            beam_width=beam_width, depth=depth, w=w,
            w_gap=w_gap, gap_mode=gap_mode,
            patience=patience,
        )
        lm = len(ml_moves)

        sum_base += lb
        sum_ml += lm

        gain = lb - lm
        if gain > 0:
            improved += 1
            total_gain_pos += gain
            if gain > max_gain:
                max_gain = gain
                max_gain_id = rid
        elif gain == 0:
            same += 1
        else:
            worse += 1

        if log_every and (i % log_every == 0 or i == 1 or i == N):
            print(f"  [{i:4d}/{N}] base={lb:3d} ml={lm:3d} gain={gain:3d}  dt={time.time()-t0:7.1f}s", flush=True)

    dt = time.time() - t0
    return {
        "baseline_total": int(sum_base),
        "ml_total": int(sum_ml),
        "total_gain": int(sum_base - sum_ml),
        "improved_cases": int(improved),
        "same_cases": int(same),
        "worse_cases": int(worse),
        "avg_gain_when_improved": float(total_gain_pos / improved) if improved else 0.0,
        "max_gain": int(max_gain),
        "max_gain_id": int(max_gain_id) if max_gain_id is not None else None,
        "time_sec_eval": float(dt),
        "sec_per_sample_eval": float(dt / max(1, N)),
        "mean_baseline_len": float(sum_base / max(1, N)),
        "mean_ml_len": float(sum_ml / max(1, N)),
        "improved_frac": float(improved / max(1, N)),
    }
