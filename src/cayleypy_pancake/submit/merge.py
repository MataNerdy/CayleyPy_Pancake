from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from cayleypy_pancake.utils.solution_format import moves_len


PathLike = Union[str, Path]


def merge_submissions_with_partials(
    *,
    base_paths: List[PathLike],
    partial_paths: List[PathLike],
    out_path: PathLike = "submission_final.csv",
    save_source_column: bool = True,
    tie_break: str = "keep_base",   # "keep_base" | "prefer_partial"
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Merge multiple full submissions (base_paths) by per-id minimal moves length,
    then apply partial submissions (partial_paths) as overrides if strictly better
    (or also on ties if tie_break=='prefer_partial').

    Expected columns: id, solution
    Returns df with columns: id, solution (+ optional source)
    """
    if tie_break not in {"keep_base", "prefer_partial"}:
        raise ValueError("tie_break must be: keep_base | prefer_partial")

    out_path = Path(out_path)

    # ----- load base submissions
    base_subs: List[pd.DataFrame] = []
    for p in base_paths:
        p = Path(p)
        df = pd.read_csv(p)
        if not {"id", "solution"}.issubset(df.columns):
            raise ValueError(f"{p} must have columns: id, solution")

        df = df[["id", "solution"]].copy()
        df["id"] = df["id"].astype(int)
        df = df.sort_values("id").reset_index(drop=True)
        df["len"] = df["solution"].map(moves_len)
        df["source"] = p.stem  # tag by file stem
        base_subs.append(df)

    if not base_subs:
        raise ValueError("base_paths is empty")

    # sanity: ids match across all bases
    base_ids = base_subs[0]["id"].values
    for i, df in enumerate(base_subs[1:], start=1):
        if len(df) != len(base_subs[0]) or not np.array_equal(df["id"].values, base_ids):
            raise ValueError(f"ID mismatch between {Path(base_paths[0])} and {Path(base_paths[i])}")

    # ----- ensemble bases by minimal len
    best = base_subs[0][["id", "solution", "len", "source"]].copy()
    for df in base_subs[1:]:
        better = df["len"].values < best["len"].values
        best.loc[better, "solution"] = df.loc[better, "solution"].values
        best.loc[better, "len"] = df.loc[better, "len"].values
        best.loc[better, "source"] = df.loc[better, "source"].values

    best["source"] = best["source"].astype(str)
    id_to_idx: Dict[int, int] = {int(pid): i for i, pid in enumerate(best["id"].values)}

    # ----- apply partials
    applied_stats: List[Tuple[str, int, int, int]] = []
    for p in partial_paths:
        p = Path(p)
        if not p.exists():
            if not quiet:
                print(f"[WARN] partial not found, skipped: {p}")
            continue

        part = pd.read_csv(p)
        if not {"id", "solution"}.issubset(part.columns):
            raise ValueError(f"{p} must have columns: id, solution")

        part = part[["id", "solution"]].copy()
        part["id"] = part["id"].astype(int)
        part["len"] = part["solution"].map(moves_len)

        replaced = 0
        missing = 0

        src_tag = p.stem.replace("submission_", "")

        for pid, sol, sol_len in part.itertuples(index=False):
            idx = id_to_idx.get(int(pid))
            if idx is None:
                missing += 1
                continue

            cur_len = int(best.at[idx, "len"])
            cur_sol = best.at[idx, "solution"]

            if (sol_len < cur_len) or (
                tie_break == "prefer_partial" and sol_len == cur_len and sol != cur_sol
            ):
                best.at[idx, "solution"] = sol
                best.at[idx, "len"] = int(sol_len)
                best.at[idx, "source"] = src_tag
                replaced += 1

        applied_stats.append((str(p), replaced, missing, len(part)))

    # ----- output
    out_df = best[["id", "solution"]].copy()
    if save_source_column:
        out_df["source"] = best["source"].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if not quiet:
        total_moves = int(best["len"].sum())
        print("\n=== MERGE SUMMARY ===")
        print("Output:", str(out_path))
        print("Rows:", len(out_df))
        print("Total moves (score):", total_moves)
        if applied_stats:
            print("\nApplied partials:")
            for p, replaced, missing, rows in applied_stats:
                print(f"  {p}: replaced={replaced} missing_ids={missing} rows={rows}")
        if save_source_column:
            print("\nWinners by source (top 20):")
            print(out_df["source"].value_counts().head(20).to_string())
        print("\nSaved:", str(out_path))

    return out_df
