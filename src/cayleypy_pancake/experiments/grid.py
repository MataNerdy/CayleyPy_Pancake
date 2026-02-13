import pandas as pd
from dataclasses import dataclass
from typing import List, Callable, Iterable, Tuple
import random
import time
import itertools

from cayleypy_pancake.baseline import parse_permutation, pancake_sort_moves
from cayleypy_pancake.search import make_h, beam_improve_or_baseline_h, apply_moves


def select_cases_per_n(
    df: pd.DataFrame,
    n_list: List[int],
    k: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for n in n_list:
        sub = df[df["n"] == n]
        if len(sub) < k:
            raise ValueError(f"Not enough samples for n={n}: have {len(sub)}, need {k}")
        idxs = list(sub.index)
        rng.shuffle(idxs)
        chosen = idxs[:k]
        rows.append(df.loc[chosen])
    return pd.concat(rows, axis=0).reset_index(drop=True)

def log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)

@dataclass
class RunRow:
    id: int
    n: int
    base_len: int
    alpha: float
    w: float
    beam_width: int
    depth: int
    ok: bool
    steps: int
    gain: int
    time_sec: float

def run_grid(
    mini_df: pd.DataFrame,
    *,
    alphas: List[float],
    ws: List[float],
    beam_widths: List[int],
    depths: List[int],
    baseline_moves_fn: Callable[[Iterable[int]], List[int]] = pancake_sort_moves,
    log: bool = True,
    log_each: int = 1,
    beam_log: bool = False,
    beam_log_every_layer: int = 5,
) -> pd.DataFrame:
    rows: List[dict] = []
    total_cfg = len(alphas) * len(ws) * len(beam_widths) * len(depths)
    total_cases = len(mini_df)
    total_runs = total_cfg * total_cases

    log_print(log, f"[grid] cases={total_cases} cfg_per_case={total_cfg} total_runs={total_runs}")
    parsed: List[Tuple[int, int, List[int], int]] = []
    for i in range(total_cases):
        row = mini_df.iloc[i]
        perm = parse_permutation(row.permutation)
        n = len(perm)
        base_len = len(baseline_moves_fn(perm))
        parsed.append((int(row.id), n, perm, base_len))

    run_idx = 0
    for case_i, (rid, n, perm, base_len) in enumerate(parsed):
        log_print(log, f"\n[case {case_i+1}/{total_cases}] id={rid} n={n} base_len={base_len}")

        for cfg_i, (alpha, w, bw, d) in enumerate(itertools.product(alphas, ws, beam_widths, depths), start=1):
            run_idx += 1
            do_cfg_log = log and (cfg_i % max(1, log_each) == 0)

            t0 = time.time()
            h_fn = make_h(alpha)
            sol = beam_improve_or_baseline_h(
                perm,
                baseline_moves_fn=baseline_moves_fn,
                h_fn=h_fn,
                beam_width=bw,
                depth=d,
                w=w,
                log=(beam_log and do_cfg_log),
                log_every_layer=beam_log_every_layer,
            )
            dt = time.time() - t0

            ok = (apply_moves(perm, sol) == list(range(n)))
            steps = len(sol)
            gain = base_len - steps

            if do_cfg_log:
                log_print(
                    True,
                    f"[run {run_idx}/{total_runs}] cfg={cfg_i:03d}/{total_cfg} "
                    f"alpha={alpha} w={w} bw={bw} depth={d} -> steps={steps} gain={gain} ok={ok} t={dt:.3f}s"
                )

            rows.append({
                "id": rid,
                "n": n,
                "base_len": base_len,
                "alpha": alpha,
                "w": w,
                "beam_width": bw,
                "depth": d,
                "ok": ok,
                "steps": steps,
                "gain": gain,
                "time_sec": dt,
            })

    df = pd.DataFrame(rows)
    return df
