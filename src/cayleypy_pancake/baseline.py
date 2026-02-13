from __future__ import annotations
from typing import Iterable, List

def parse_permutation(raw: str) -> List[int]:
    """Parse CSV cell like '0,4,2,5,3,1' into list[int]."""
    if raw is None:
        return []
    raw = str(raw).strip()
    if raw == "":
        return []
    return [int(tok) for tok in raw.split(",") if tok.strip() != ""]

def pancake_sort_moves(perm: Iterable[int]) -> List[int]:
    """
    Classic pancake sorting baseline (no burnt pancakes):
    returns flip sizes k (meaning reverse prefix of length k).
    """
    a = list(perm)
    n = len(a)
    if n <= 1:
        return []

    pos = [0] * n
    for i, v in enumerate(a):
        pos[v] = i

    moves: List[int] = []

    def do_flip(k: int) -> None:
        if k <= 1:
            return
        i, j = 0, k - 1
        while i < j:
            vi, vj = a[i], a[j]
            a[i], a[j] = vj, vi
            pos[vi], pos[vj] = j, i
            i += 1
            j -= 1

    for target in range(n - 1, 0, -1):
        idx = pos[target]
        if idx == target:
            continue
        if idx != 0:
            do_flip(idx + 1)
            moves.append(idx + 1)
        do_flip(target + 1)
        moves.append(target + 1)

    return moves
