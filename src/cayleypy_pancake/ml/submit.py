from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from cayleypy_pancake.baseline import parse_permutation, pancake_sort_moves
from cayleypy_pancake.models import get_model
from cayleypy_pancake.ml.train import train_model_gpu
from cayleypy_pancake.ml.heuristic import MLHeuristic
from cayleypy_pancake.ml.search import beam_improve_with_ml
from cayleypy_pancake.utils.logging import now_str
from cayleypy_pancake.utils.solution_format import moves_to_str, moves_len
from cayleypy_pancake.utils.progress import load_progress_map, save_progress_map


def build_submission_with_ml(
    *,
    test_df: pd.DataFrame,
    n_list: List[int],
    leader_cfg: Dict,
    device: Optional[str] = None,
    baseline_moves_fn=pancake_sort_moves,
    beam_width: int = 256,
    depth: int = 192,
    w: float = 0.5,
    w_gap: float = 0.15,
    gap_mode: str = "log1p",
    h_batch_size: int = 8192,
    baseline_path: Path,
    progress_path: Path,
    final_path: Path,
    flush_every: int = 200,
) -> pd.DataFrame:
    """
    Train per-n ML heuristic (from leader_cfg) and improve baseline via beam search.
    Uses progress_path for resume, writes final submission to final_path.
    """
    from cayleypy_pancake.ml.graph import build_pancake_graph
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load baseline/progress maps
    baseline_map = load_progress_map(Path(baseline_path))
    progress_map = load_progress_map(Path(progress_path))
    if progress_map:
        print("Resume progress:", len(progress_map), flush=True)
    else:
        print("No progress, start fresh.", flush=True)

    t_global = time.time()
    done_cnt = 0
    improved_cnt = 0

    # Ensure test_df has column n
    if "n" not in test_df.columns:
        tmp = test_df.copy()
        tmp["n"] = tmp["permutation"].apply(lambda x: len(parse_permutation(x)))
        test_df = tmp

    for n in n_list:
        print(f"\n========== TARGET n = {n} ==========", flush=True)

        graph = build_pancake_graph(n, device=device)

        CFG = dict(leader_cfg)
        CFG["n"] = n
        CFG["num_classes"] = n
        CFG["state_size"] = n

        k = int(CFG.get("k", 4))
        rw_length_add = int(CFG.get("rw_length_add", 30))
        CFG["rw_length"] = n * (n + 5) // (4 * (k - 1)) + rw_length_add

        model = get_model(CFG).to(graph.device)

        print(
            f"[{now_str()}] train model for n={n} | epochs={CFG['num_epochs']} "
            f"rw_width={CFG['rw_width']} mix={CFG.get('mix_bfs_frac')}",
            flush=True,
        )
        t_train0 = time.time()
        train_model_gpu(CFG, model, graph)
        print(f"train_time={time.time()-t_train0:.1f}s", flush=True)

        h_ml = MLHeuristic(model, device=graph.device, batch_size=h_batch_size)

        sub = test_df[test_df["n"] == n][["id", "permutation"]].reset_index(drop=True)
        total_n = len(sub)
        print("cases:", total_n, flush=True)

        t0 = time.time()
        for i in range(total_n):
            pid = int(sub.loc[i, "id"])

            base_str = baseline_map.get(pid, "")
            if base_str == "":
                perm0 = parse_permutation(sub.loc[i, "permutation"])
                base_str = moves_to_str(baseline_moves_fn(perm0))
                baseline_map[pid] = base_str

            perm = parse_permutation(sub.loc[i, "permutation"])
            base_len = moves_len(base_str)

            moves = beam_improve_with_ml(
                perm,
                h_fn=h_ml,
                baseline_moves_fn=baseline_moves_fn,
                beam_width=beam_width,
                depth=depth,
                w=w,
                w_gap=w_gap,
                gap_mode=gap_mode,
                patience=None,
            )
            new_str = moves_to_str(moves)
            new_len = len(moves)

            if new_len < base_len:
                progress_map[pid] = new_str
                improved_cnt += 1
            else:
                progress_map[pid] = base_str

            done_cnt += 1

            if flush_every and ((i + 1) % flush_every == 0 or (i + 1) == total_n):
                save_progress_map(progress_map, Path(progress_path))
                dt = time.time() - t0
                print(
                    f"  [{i+1:5d}/{total_n}] improved={improved_cnt} done_total={done_cnt} "
                    f"dt_n={dt:6.1f}s dt_all={time.time()-t_global:7.1f}s",
                    flush=True,
                )

        # cleanup per-n
        try:
            graph.free_memory()
        except Exception:
            pass
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final submission
    final_df = (
        pd.DataFrame(list(progress_map.items()), columns=["id", "solution"])
        .sort_values("id")
        .reset_index(drop=True)
    )
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(Path(final_path), index=False)

    # Save progress as well (latest snapshot)
    save_progress_map(progress_map, Path(progress_path))

    print(f"\nSaved: {final_path} rows={len(final_df)}", flush=True)
    print(f"Progress saved: {progress_path}", flush=True)
    return final_df
