from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd


def load_progress_map(path: Path) -> Dict[int, str]:
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["id"].astype(int).values, df["solution"].astype(str).values))


def save_progress_map(progress_map: Dict[int, str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(progress_map.items()), columns=["id", "solution"]).sort_values("id")
    df.to_csv(path, index=False)
