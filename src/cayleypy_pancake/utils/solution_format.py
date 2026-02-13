from __future__ import annotations
from typing import List
import math

def moves_to_str(moves: List[int]) -> str:
    """Encode move list [k1, k2, ...] as 'Rk1.Rk2....'."""
    return ".".join(f"R{k}" for k in moves)

def moves_len(sol) -> int:
    """Length of encoded solution string 'R3.R5....'. Returns 0 for empty/NaN/None."""
    if sol is None:
        return 0
    if isinstance(sol, float) and math.isnan(sol):
        return 0
    s = str(sol).strip()
    if s == "":
        return 0
    return s.count(".") + 1
