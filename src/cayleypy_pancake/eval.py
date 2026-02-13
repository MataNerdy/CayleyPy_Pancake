from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Callable
import time
import pandas as pd

from cayleypy_pancake.baseline import parse_permutation, pancake_sort_moves
from cayleypy_pancake.search import (
    make_h,
    beam_improve_or_baseline_h,
    apply_moves,
    log_print,
)
from cayleypy_pancake.utils.solution_format import moves_to_str


def full_eval_top_cfgs(
    test_df: pd.DataFrame,
    n_list: List[int],
    top_cfgs: List[Dict[str, Any]],
    *,
    out_csv_path: Path,
    baseline_moves_fn: Callable = pancake_sort_moves,
    log: bool = True,
    log_every: int = 50,
    flush_every: int = 200,
) -> pd.DataFrame:
    """
    Evaluate top configurations on subset of test_df where n in n_list.
    Appends results into out_csv_path with resume support by (id, cfg_idx).
    Writes in chunks to avoid losing progress.
    Returns the whole CSV as DataFrame (reads back from disk).
    """
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sub = test_df[test_df["n"].isin(n_list)].reset_index(drop=True)

    done = set()
    wrote_header = out_csv_path.exists()
    if wrote_header:
        try:
            prev = pd.read_csv(out_csv_path, usecols=["id", "cfg_idx"])
            done = set(zip(prev["id"].astype(int).tolist(), prev["cfg_idx"].astype(int).tolist()))
            if log:
                log_print(True, f"[resume] found existing file with {len(done)} completed runs")
        except Exception as e:
            if log:
                log_print(True, f"[resume] could not read existing file safely: {e!r} (will append anyway)")

    rows_cache: List[Dict[str, Any]] = []

    t_global0 = time.time()
    total_cases = len(sub)
    total_runs = total_cases * len(top_cfgs)

    run_idx = 0
    skipped = 0

    for i in range(total_cases):
        rid = int(sub.loc[i, "id"])
        perm = parse_permutation(sub.loc[i, "permutation"])
        n = len(perm)

        base_moves = baseline_moves_fn(perm)
        base_len = len(base_moves)

        for cfg_j, cfg in enumerate(top_cfgs):
            run_idx += 1

            if (rid, cfg_j) in done:
                skipped += 1
                if log and (run_idx % log_every == 0):
                    elapsed = time.time() - t_global0
                    speed = (run_idx - skipped) / max(1e-9, elapsed)
                    log_print(True, f"[full] {run_idx}/{total_runs} runs | skipped={skipped} | speed={speed:.3f} new_runs/s")
                continue

            alpha = float(cfg["alpha"])
            w = float(cfg["w"])
            bw = int(cfg["beam_width"])
            depth = int(cfg["depth"])

            t0 = time.time()
            status = "ok"
            err_txt = ""

            try:
                h_fn = make_h(alpha)
                moves = beam_improve_or_baseline_h(
                    perm,
                    baseline_moves_fn=baseline_moves_fn,
                    h_fn=h_fn,
                    beam_width=bw,
                    depth=depth,
                    w=w,
                    log=False,
                )

                if apply_moves(perm, moves) != list(range(n)):
                    moves = base_moves
                    status = "fallback_baseline"

            except Exception as e:
                moves = base_moves
                status = "error_fallback_baseline"
                err_txt = repr(e)

            dt = time.time() - t0
            steps = len(moves)
            gain = base_len - steps

            row = {
                "id": rid,
                "n": n,
                "cfg_idx": cfg_j,
                "alpha": alpha,
                "w": w,
                "beam_width": bw,
                "depth": depth,
                "base_len": base_len,
                "ok": (status == "ok"),
                "steps": steps,
                "gain": gain,
                "time_sec": dt,
                "solution": moves_to_str(moves),
                "status": status,
                "error": err_txt,
            }

            rows_cache.append(row)
            done.add((rid, cfg_j))

            if len(rows_cache) >= flush_every:
                pd.DataFrame(rows_cache).to_csv(
                    out_csv_path,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
                rows_cache.clear()

            if log and (run_idx % log_every == 0):
                elapsed = time.time() - t_global0
                new_done = run_idx - skipped
                speed = new_done / max(1e-9, elapsed)
                log_print(
                    True,
                    f"[full] {run_idx}/{total_runs} runs | new={new_done} skipped={skipped} | "
                    f"n={n} cfg={cfg_j} steps={steps} gain={gain} status={status} | {speed:.3f} new_runs/s"
                )

    if rows_cache:
        pd.DataFrame(rows_cache).to_csv(
            out_csv_path,
            mode="a",
            header=not wrote_header,
            index=False,
        )

    return pd.read_csv(out_csv_path)
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Callable
import time
import pandas as pd

from cayleypy_pancake.baseline import parse_permutation, pancake_sort_moves
from cayleypy_pancake.utils.solution_format import moves_len


def evaluate_submission_vs_baseline(
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    *,
    baseline_moves_fn: Callable = pancake_sort_moves,
    log_every: int = 0,
    save_detailed_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Compare a submission (id, solution) against baseline lengths on the same test_df.
    Does NOT verify correctness of the submission moves; it only compares lengths.
    Optionally saves per-id detailed rows.
    """
    t0 = time.time()
    sub_map = dict(zip(submission_df["id"].astype(int), submission_df["solution"].astype(str)))

    sum_base = 0
    sum_sub = 0
    improved = same = worse = 0
    total_gain_pos = 0
    max_gain = 0
    max_gain_id = None

    N = len(test_df)
    detailed_rows = [] if save_detailed_path else None

    for i, row in enumerate(test_df.itertuples(index=False), start=1):
        rid = int(row.id)
        perm = parse_permutation(row.permutation)

        base = baseline_moves_fn(perm)
        lb = len(base)

        sol = sub_map.get(rid, "")
        lz = moves_len(sol)

        sum_base += lb
        sum_sub += lz

        gain = lb - lz
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

        if detailed_rows is not None:
            detailed_rows.append({
                "id": rid,
                "n": len(perm),
                "base_len": lb,
                "sub_len": lz,
                "gain": gain,
            })

        if log_every and (i % log_every == 0 or i == 1 or i == N):
            print(
                f"[{i:4d}/{N}] base={lb:3d} sub={lz:3d} gain={gain:3d}  elapsed={time.time()-t0:7.1f}s",
                flush=True,
            )

    dt = time.time() - t0

    if save_detailed_path:
        save_detailed_path = Path(save_detailed_path)
        save_detailed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detailed_rows).to_csv(save_detailed_path, index=False)

    return {
        "baseline_total": sum_base,
        "submission_total": sum_sub,
        "total_gain": (sum_base - sum_sub),
        "improved_cases": improved,
        "same_cases": same,
        "worse_cases": worse,
        "avg_gain_when_improved": (total_gain_pos / improved) if improved else 0.0,
        "max_gain": max_gain,
        "max_gain_id": max_gain_id,
        "time_sec": dt,
        "sec_per_sample": dt / max(1, N),
        "mean_baseline_len": sum_base / max(1, N),
        "mean_submission_len": sum_sub / max(1, N),
        "improved_frac": improved / max(1, N),
    }
