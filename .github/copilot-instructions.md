Project: CayleyPy_Pancake — guidance for AI coding agents

Overview
- This repo contains research code and notebooks for pancake sorting on Cayley graphs.
- Core package: `src/cayleypy_pancake` implements models, search, baselines, and evaluation utilities used by notebooks in `notebooks/`.

What to prioritize
- Preserve backward-compatibility with the notebooks. Notebooks call functions in `cayleypy_pancake.*` and expect simple, stable APIs (e.g. `get_model(cfg)` in `models.py`).
- Prefer adding small, well-scoped helpers under `src/cayleypy_pancake/utils/` when you need shared code.

Architecture & data flow (quick)
- Input data: CSVs in `data/` (e.g. `test.csv`) containing columns `id`, `permutation`, `n`.
- Baseline moves: `baseline.pancake_sort_moves(perm)` returns list of flip sizes k. Parsers: `baseline.parse_permutation`.
- Heuristic search: `search.beam_improve_or_baseline_h(perm, baseline_moves_fn=..., h_fn=..., beam_width=..., depth=..., w=...)` returns moves list or falls back to baseline.
- Evaluation: `eval.full_eval_top_cfgs(...)` runs many configs and appends results to CSV; `eval.evaluate_submission_vs_baseline(...)` compares submission lengths to baseline.
- Models: `models.get_model(cfg)` chooses between `EmbMLP`, `Pilgrim` (MLP Res), and `SimpleMLP`. Models expect integer tensors representing permutations (shape: B x n).

Important files to inspect when changing behavior
- `src/cayleypy_pancake/models.py` — model constructors and forward signatures. Keep `cfg` keys consistent (e.g. `state_size`, `num_classes`, `emb_dim`, `hd1`, `hd2`, `nrd`).
- `src/cayleypy_pancake/search.py` — beam search and heuristics. Logging via `log_print(enabled,msg)`; careful when changing return types (expects list[int] or None).
- `src/cayleypy_pancake/baseline.py` — canonical pancake sorting implementation used as fallback.
- `src/cayleypy_pancake/eval.py` — orchestrates large runs and writes CSVs; respects resume by checking existing CSV output (columns `id`, `cfg_idx`).
- `src/cayleypy_pancake/utils/solution_format.py` and `utils/moves.py` — canonical encoding for solutions: 'Rk.Rj...' and helpers `moves_len`, `moves_to_str` used across code.

Conventions & patterns
- Solutions are encoded as strings of dot-separated prefix flips, e.g. `R3.R5.R2`. Use `moves_to_str` / `moves_len` for encoding/lengths.
- Parsers accept empty or NaN permutation cells and return empty lists — replicate this behavior in new code.
- Functions often accept either iterables or lists for permutations; keep accepting general iterables where practical.
- Logging: use the small `log_print(enabled, msg)` pattern used in `search.py` and `eval.py` to avoid noisy prints; prefer keeping `log` flags in public functions.

Developer workflows & commands
- Packaging: this is a Python project with `pyproject.toml` (setuptools). No heavy build steps.
- Typical quick dev steps (run from repo root):
  - Run a notebook in `notebooks/` (Jupyter) — notebooks import package via `src` layout. Use an editable install for iteration: `pip install -e .` in a virtualenv with Python >= 3.10.
  - Run small scripts: import `cayleypy_pancake` in REPL to test functions (no tests present).

Edge cases and pitfalls
- CSV resume semantics: `eval.full_eval_top_cfgs` reads existing out CSV and uses pairs `(id, cfg_idx)` to resume; avoid changing column names or types.
- `models` expect `state_size == num_classes` for `EmbMLP` (asserted). Don't silently change that assumption.
- Search functions assume flips k in range 2..n inclusive. Keep that domain when emitting moves.

When adding tests or examples
- Add compact unit tests under a new `tests/` folder if needed. Tests should target small inputs (n<=6) to keep runs fast.
- Add minimal examples called from `notebooks/` or a `scripts/` helper that reproduce common runs (e.g., one-shot eval on a tiny subset of `data/test.csv`).

If you change public APIs
- Update `eval.py` and notebooks references together. Notebooks are the primary integration tests — keep them runnable.

Contact points in code (search targets)
- `get_model(`, `pancake_sort_moves(`, `beam_improve_or_baseline_h(`, `full_eval_top_cfgs(`, `moves_to_str(`, `moves_len(`, `parse_permutation(`

If anything above is unclear or you want examples (unit tests, small scripts, or a compact notebook cell) added, tell me which area and I'll iterate.
