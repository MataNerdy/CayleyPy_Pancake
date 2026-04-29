from cayleypy_pancake.baseline import pancake_sort_moves
from cayleypy_pancake.search import beam_improve_or_baseline_h, make_h, apply_moves
from cayleypy_pancake.utils.solution_format import moves_to_str


def main():
    perm = [9,1,7,11,4,3,0,8,5,2,10,6]

    baseline_moves = pancake_sort_moves(perm)

    beam_moves = beam_improve_or_baseline_h(
        perm,
        baseline_moves_fn=pancake_sort_moves,
        h_fn=make_h(alpha=0.0),   # gap heuristic
        beam_width=128,
        depth=128,
        w=0.5,
        log=False,
    )

    baseline_final = apply_moves(perm, baseline_moves)
    beam_final = apply_moves(perm, beam_moves)

    print("🥞 Pancake Sorting Demo")
    print("-" * 50)

    print("Start permutation:", perm)
    print()

    print("Baseline")
    print("  moves:", moves_to_str(baseline_moves))
    print("  length:", len(baseline_moves))
    print("  solved:", baseline_final == sorted(perm))
    print()

    print("Beam search improvement")
    print("  moves:", moves_to_str(beam_moves))
    print("  length:", len(beam_moves))
    print("  solved:", beam_final == sorted(perm))
    print()

    print("Gain:", len(baseline_moves) - len(beam_moves))


if __name__ == "__main__":
    main()