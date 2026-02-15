from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from cayleypy_pancake.baseline import pancake_sort_moves
from cayleypy_pancake.models import get_model
from cayleypy_pancake.utils.logging import now_str
from cayleypy_pancake.utils.naming import exp_name_from
from cayleypy_pancake.utils.seed import set_seed  # проверь имя в твоём utils/seed.py

from cayleypy_pancake.ml.graph import build_pancake_graph
from cayleypy_pancake.ml.train import train_model_gpu
from cayleypy_pancake.ml.metrics import sanity_corr_bfs
from cayleypy_pancake.ml.heuristic import MLHeuristic
from cayleypy_pancake.ml.eval import build_rows_for_n, eval_ml_on_rows


def run_one_experiment_cached(
    *,
    rows,
    graph,
    target_n: int,
    cfg: dict,
    exp_name: str,
    out_dir: Path,
    baseline_moves_fn=pancake_sort_moves,
    beam_width: int = 256,
    depth: int = 192,
    w: float = 0.5,
    w_gap: float = 0.15,
    gap_mode: str = "log1p",
    patience: Optional[int] = None,
) -> dict:
    set_seed(int(cfg.get("seed", 123)))

    CFG = dict(cfg)
    CFG["n"] = target_n
    CFG["num_classes"] = target_n
    CFG["state_size"] = target_n

    k = int(CFG.get("k", 4))
    rw_length_add = int(CFG.get("rw_length_add", 30))
    CFG["rw_length"] = target_n * (target_n + 5) // (4 * (k - 1)) + rw_length_add

    model = get_model(CFG).to(graph.device)

    print(
        f"\n[{now_str()}] EXP={exp_name} | model={CFG.get('model_type')} n={target_n} "
        f"| epochs={CFG['num_epochs']} rw_width={CFG['rw_width']} rw_mode={CFG.get('rw_mode')}",
        flush=True,
    )

    t_train0 = time.time()
    train_model_gpu(CFG, model, graph)
    train_time = time.time() - t_train0

    try:
        corr = sanity_corr_bfs(
            model, graph,
            rw_length=CFG["rw_length"],
            width=int(CFG.get("sanity_width", 2000)),
            nbt_history_depth=int(CFG.get("nbt_history_depth", 1)),
        )
    except Exception as e:
        corr = float("nan")
        print("[sanity] failed:", repr(e), flush=True)

    h_ml = MLHeuristic(model, device=graph.device, batch_size=int(CFG.get("h_batch_size", 8192)))

    eval_stats = eval_ml_on_rows(
        rows,
        h_ml=h_ml,
        baseline_moves_fn=baseline_moves_fn,
        beam_width=beam_width, depth=depth, w=w,
        w_gap=w_gap, gap_mode=gap_mode,
        patience=patience,
        log_every=int(CFG.get("eval_log_every", 10)),
    )

    res = {
        "exp_name": exp_name,
        "timestamp": now_str(),
        "target_n": int(target_n),
        "rows_k": int(len(rows)),

        "model_type": str(CFG.get("model_type")),
        "emb_dim": int(CFG["emb_dim"]) if CFG.get("model_type") == "EmbMLP" else None,
        "use_pos_emb": bool(CFG.get("use_pos_emb", True)) if CFG.get("model_type") == "EmbMLP" else None,
        "hd1": int(CFG.get("hd1", 0)),
        "hd2": int(CFG.get("hd2", 0)),
        "nrd": int(CFG.get("nrd", 0)),
        "dropout_rate": float(CFG.get("dropout_rate", 0.0)),

        "rw_width": int(CFG.get("rw_width", 0)),
        "rw_length": int(CFG.get("rw_length", 0)),
        "rw_mode": str(CFG.get("rw_mode", "")),
        "mix_bfs_frac": float(CFG.get("mix_bfs_frac", 0.0)),
        "nbt_history_depth": int(CFG.get("nbt_history_depth", 1)),
        "lr": float(CFG.get("lr", 0.0)),
        "weight_decay": float(CFG.get("weight_decay", 0.0)),
        "batch_size": int(CFG.get("batch_size", 0)),
        "val_ratio": float(CFG.get("val_ratio", 0.0)),
        "num_epochs": int(CFG.get("num_epochs", 0)),
        "y_transform": str(CFG.get("y_transform", "")),
        "loss_beta": float(CFG.get("loss_beta", 1.0)),
        "grad_clip": float(CFG.get("grad_clip", 0.0)),

        "beam_width": int(beam_width),
        "depth": int(depth),
        "w": float(w),
        "w_gap": float(w_gap),
        "gap_mode": str(gap_mode),
        "patience": patience if patience is None else int(patience),

        "train_time_sec": float(train_time),
        "sanity_corr": float(corr),
        **eval_stats,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{exp_name}.json", "w") as f:
        json.dump(res, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def run_sweep_n_list(
    *,
    test_df: pd.DataFrame,
    n_list: list[int],
    rows_k: int = 50,
    rows_seed: int = 42,
    model_grid: list[dict],
    w_gap: float = 0.15,
    gap_mode: str = "log1p",
    results_csv: Path = Path("ml_sweep/results_nlist.csv"),
    out_json_dir: Path = Path("ml_sweep/json"),
    device: str | None = None,
    beam_width: int = 256,
    depth: int = 192,
    w: float = 0.5,
) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results_csv = Path(results_csv)
    out_json_dir = Path(out_json_dir)
    out_json_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if results_csv.exists():
        prev = pd.read_csv(results_csv, usecols=["exp_name"])
        done = set(prev["exp_name"].astype(str))
        print(f"[resume] {len(done)} experiments already done", flush=True)

    rows_cache = {n: build_rows_for_n(test_df, n, k=rows_k, seed=rows_seed) for n in n_list}

    total = len(n_list) * len(model_grid)
    run_i = 0

    try:
        for n in n_list:
            print(f"\n########## TARGET_N = {n} ##########", flush=True)

            graph = build_pancake_graph(n, device=device)
            rows = rows_cache[n]

            best_gain = -10**9
            best_name = None

            for cfg in model_grid:
                run_i += 1
                exp_name = exp_name_from(cfg, n=n, w_gap=w_gap, gap_mode=gap_mode)

                if exp_name in done:
                    print(f"[skip] {run_i}/{total} {exp_name}", flush=True)
                    continue

                print(f"\n===== [{run_i}/{total}] {exp_name} =====", flush=True)

                res = run_one_experiment_cached(
                    rows=rows,
                    graph=graph,
                    target_n=n,
                    cfg=cfg,
                    exp_name=exp_name,
                    out_dir=out_json_dir,
                    baseline_moves_fn=pancake_sort_moves,
                    beam_width=beam_width,
                    depth=depth,
                    w=w,
                    w_gap=w_gap,
                    gap_mode=gap_mode,
                    patience=None,
                )

                df1 = pd.DataFrame([res])
                header = not results_csv.exists()
                df1.to_csv(results_csv, mode="a", header=header, index=False)

                done.add(exp_name)

                if res["total_gain"] > best_gain:
                    best_gain = res["total_gain"]
                    best_name = exp_name

                print(f"[best@n={n}] total_gain={best_gain} ({best_name})", flush=True)

            # освобождаем граф после n
            try:
                graph.free_memory()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.read_csv(results_csv)
