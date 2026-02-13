from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd

def load_progress_map(progress_path: Path) -> Dict[int, str]:
    """
    CSV format: columns ['id', 'solution'] where id is int-like.
    Returns dict: id -> solution.
    """
    if progress_path.exists():
        df = pd.read_csv(progress_path)
        mp = dict(zip(df["id"].astype(int).values, df["solution"].values))
        return mp
    return {}

def save_progress_map(progress_path: Path, progress_map: Dict[int, str]) -> None:
    """
    Writes progress_map back to CSV with columns ['id', 'solution'].
    """
    df = pd.DataFrame({"id": list(progress_map.keys()), "solution": list(progress_map.values())})
    df.to_csv(progress_path, index=False)
