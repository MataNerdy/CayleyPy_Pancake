import streamlit as st

from cayleypy_pancake.baseline import pancake_sort_moves
from cayleypy_pancake.search import beam_improve_or_baseline_h, make_h, apply_moves
from cayleypy_pancake.utils.solution_format import moves_to_str


def parse_input(input_str: str) -> list[int]:
    return [int(x.strip()) for x in input_str.split(",") if x.strip()]


def path_to_states(start: list[int], moves: list[int]) -> list[list[int]]:
    states = [list(start)]
    cur = list(start)

    for k in moves:
        cur = list(cur)
        cur[:k] = reversed(cur[:k])
        states.append(list(cur))

    return states


def is_solved(state: list[int]) -> bool:
    return state == list(range(len(state)))


def gap_h(state: list[int]) -> int:
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


def breakpoints(state: list[int]) -> int:
    n = len(state)
    b = 0

    for i in range(n - 1):
        if abs(state[i] - state[i + 1]) != 1:
            b += 1

    if n > 0 and state[0] != 0:
        b += 1

    return b


st.set_page_config(page_title="Pancake Solver", layout="centered")

st.title("🥞 Pancake Solver")
st.caption("Baseline pancake sort vs heuristic beam search")

input_str = st.text_input(
    "Введите перестановку через запятую",
    "9,1,7,11,4,3,0,8,5,2,10,6",
)

beam_width = st.slider("Beam width", 1, 512, 128)
depth = st.slider("Depth", 1, 256, 128)
w = st.slider("Heuristic weight w", 0.1, 2.0, 0.5, 0.1)

if st.button("Solve"):
    try:
        start = parse_input(input_str)

        if sorted(start) != list(range(len(start))):
            st.error("Перестановка должна содержать числа от 0 до n-1 без повторов.")
            st.stop()

        baseline_moves = pancake_sort_moves(start)

        beam_moves = beam_improve_or_baseline_h(
            start,
            baseline_moves_fn=pancake_sort_moves,
            h_fn=make_h(alpha=0.0),
            beam_width=beam_width,
            depth=depth,
            w=w,
            log=False,
        )

        baseline_final = apply_moves(start, baseline_moves)
        beam_final = apply_moves(start, beam_moves)

        st.subheader("Baseline")
        st.write("Solved:", is_solved(baseline_final))
        st.write("Path:", moves_to_str(baseline_moves))
        st.write("Path length:", len(baseline_moves))

        st.subheader("Beam search improvement")
        st.write("Solved:", is_solved(beam_final))
        st.write("Path:", moves_to_str(beam_moves))
        st.write("Path length:", len(beam_moves))

        gain = len(baseline_moves) - len(beam_moves)

        st.metric("Gain", gain)

        st.subheader("Состояния beam search")

        states = path_to_states(start, beam_moves)

        for i, s in enumerate(states):
            st.write(
                f"{i}: {s}, "
                f"gap={gap_h(s)}, "
                f"bp={breakpoints(s)}"
            )

    except Exception as e:
        st.error(f"Ошибка: {e}")