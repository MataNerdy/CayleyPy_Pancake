from typing import List, Iterable, Tuple, Dict, Optional, Callable
from heapq import nsmallest

def breakpoints2(state: List[int]) -> int:
    n = len(state)
    b = 0
    for i in range(n - 1):
        if abs(state[i] - state[i + 1]) != 1:
            b += 1
    if n > 0 and state[0] != 0:
        b += 1
    return b

def gap_h(state: List[int]) -> int:
    n = len(state)
    prev = -1
    gaps = 0
    for x in state:
        if abs(x - prev) != 1:
            gaps += 1
        prev = x
    if abs(n - prev) != 1:
        gaps += 1
    return gaps

def mix_h(state: List[int], alpha: float = 0.5) -> float:
    return gap_h(state) + alpha * breakpoints2(state)

def make_h(alpha: float) -> Callable[[List[int]], float]:
    if alpha == 0.0:
        return lambda s: float(gap_h(s))
    return lambda s: float(mix_h(s, alpha=alpha))

def is_solved(state: List[int]) -> bool:
    return all(v == i for i, v in enumerate(state))

def apply_move_copy(state: List[int], k: int) -> List[int]:
    nxt = state[:]
    nxt[:k] = reversed(nxt[:k])
    return nxt

def apply_moves(perm: List[int], moves: List[int]) -> List[int]:
    a = perm[:]
    for k in moves:
        a[:k] = reversed(a[:k])
    return a

def beam_improve_or_baseline_h(
    perm: Iterable[int],
    *,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    h_fn: Callable[[List[int]], float],
    beam_width: int = 8,
    depth: int = 12,
    w: float = 1.0,
    log: bool = False,
    log_every_layer: int = 1,
) -> List[int]:
    start = list(perm)

    base_moves = baseline_moves_fn(start)
    best_len = len(base_moves)
    if best_len <= 1:
        return base_moves

    apply_move = apply_move_copy
    solved = is_solved
    h_local = h_fn
    w_local = w
    k_values = range(2, len(start) + 1)

    beam: List[Tuple[float, int, List[int], List[int]]] = [
        (w_local * float(h_local(start)), 0, start, [])
    ]

    best_path: Optional[List[int]] = None
    best_g: Dict[Tuple[int, ...], int] = {tuple(start): 0}

    log_print(log, f"[beam] start n={len(start)} base_len={best_len} bw={beam_width} depth={depth} w={w}")

    for layer in range(1, depth + 1):
        candidates: List[Tuple[float, int, List[int], List[int]]] = []
        improved_this_layer = 0
        for f, g, state, path in beam:
            if g >= best_len:
                continue

            for k in k_values:
                new_g = g + 1
                if new_g >= best_len:
                    continue

                nxt = apply_move(state, k)
                key = tuple(nxt)

                prevg = best_g.get(key)
                if prevg is not None and prevg <= new_g:
                    continue
                best_g[key] = new_g

                if solved(nxt):
                    best_len = new_g
                    best_path = path + [k]
                    improved_this_layer += 1
                    continue

                h = float(h_local(nxt))
                new_f = new_g + w_local * h
                if new_f < best_len:
                    candidates.append((new_f, new_g, nxt, path + [k]))

        if log and (layer % max(1, log_every_layer) == 0):
            log_print(
                True,
                f"[beam] layer={layer:03d} beam_in={len(beam)} cand={len(candidates)} "
                f"improved={improved_this_layer} best_len={best_len}"
            )

        if not candidates:
            break

        beam = nsmallest(beam_width, candidates, key=lambda x: x[0])
        if best_len <= 2:
            break

    return best_path if best_path is not None else base_moves

def log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)