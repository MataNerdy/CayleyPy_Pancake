
from typing import List
import pandas as pd

def moves_to_str(moves: List[int]) -> str:
    return ".".join(f"R{k}" for k in moves)

def moves_len(sol) -> int:
    if sol is None or (isinstance(sol, float) and pd.isna(sol)):
        return 0
    s = str(sol).strip()
    if s == "":
        return 0
    return s.count(".") + 1