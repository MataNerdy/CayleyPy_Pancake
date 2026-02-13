from typing import List

def parse_permutation(raw: str) -> List[int]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if raw == "":
        return []
    return [int(tok) for tok in raw.split(",") if tok.strip() != ""]

def apply_move_copy(state: List[int], k: int) -> List[int]:
    nxt = state[:]
    nxt[:k] = reversed(nxt[:k])
    return nxt

def apply_moves(perm: List[int], moves: List[int]) -> List[int]:
    a = perm[:]
    for k in moves:
        a[:k] = reversed(a[:k])
    return a