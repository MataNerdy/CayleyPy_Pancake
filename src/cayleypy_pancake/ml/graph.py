from __future__ import annotations
import torch


def build_pancake_graph(target_n: int, device: str):
    from cayleypy import PermutationGroups, CayleyGraph
    central_state = list(range(target_n))
    graph = CayleyGraph(
        PermutationGroups.pancake(target_n).make_inverse_closed().with_central_state(central_state),
        device=device,
        dtype=torch.int8,
        batch_size=2**16,
    )
    return graph
