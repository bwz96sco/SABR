# Repository Guidelines

## Project Structure & Module Organization
- `ABRBench/`: dataset root (must exist at `./ABRBench`).
- `build_env_c_plus/`: C++ simulators and scripts (`build_all.sh`, `config.h`).
- `sim_env/`: Python env wrappers (`abr_gym_env.py`, `train_env.py`).
- `rl/`: learning algorithms and helpers (IL/A3C utilities).
- Root scripts: `train_sabr.py`, `train_comyco.py`, `train_pensieve.py`, `run_bs_mpc.py`, `run_rmpc_c_version.py`, `plot_result.py`, `ex_rule_baseline.sh`.
- Results/logs: created under `./test_results/…` per `config.py`.

## Build, Test, and Development Commands
- Python via `uv` (3.10 pinned by `.python-version`):
  - Ensure 3.10: `uv python install 3.10`
  - Create venv: `uv venv --python 3.10 .venv`
  - Install deps: `uv pip install -r requirements.txt`
- Configure dataset: edit `_DATASET` in `config.py` and `DATASET_OPTION` in `build_env_c_plus/config.h`.
- Build C++ env: `cd build_env_c_plus && bash build_all.sh` (rerun after any `config.h` change).
- Train SABR: `uv run train_sabr.py` (auto-evaluates on test/OOD suites).
- Baselines (learning): `uv run train_comyco.py`, `uv run train_pensieve.py`.
- Baselines (rule-based batch): `bash ex_rule_baseline.sh` (C++ tools unaffected by `uv`).
- Plot QoE: `uv run plot_result.py` (configure `SCHEMES` in `plot_result.py`).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, `snake_case` for files/functions, `CamelCase` for classes.
- Scripts follow `train_*.py`, `test_*.py`, `run_*.py` naming; keep consistent.
- Keep paths relative to repo root; avoid hard-coded absolute paths.

## Testing Guidelines
- This repo uses script-based tests (no pytest): run `test_*.py` scripts and training/eval entrypoints.
- Ensure `ABRBench` paths resolve and `test_results/` is populated.
- Prefer deterministic runs when possible (set seeds in new code; document nondeterminism).
- Validate QoE via `plot_result.py` before proposing changes.

## Commit & Pull Request Guidelines
- Commits: imperative, concise summaries (e.g., "train: fix rollout logging"). Reference issues when relevant.
- PRs: include purpose, key changes, how to run (commands), and sample results/plots. Link related issues.
- Touch both `config.py` and `build_env_c_plus/config.h` when changing datasets/QoE; describe the alignment in the PR.

## Security & Configuration Tips
- Dataset name must match: `config.py::_DATASET` ↔ `config.h::DATASET_OPTION`; rebuild after edits.
- `LOG_FILE_DIR` is auto-created by `config.py`—commit code, not generated outputs.
- `dp_my` may fail on some traces; note skips in PRs if applicable.

## macOS Build Notes
- The build scripts prefer `.venv/bin/python` and add `-undefined dynamic_lookup` on macOS to resolve Python symbols.
- Ensure `pybind11` is installed in the active env: `uv pip install -r requirements.txt`.
- Override the Python used if needed: `PYTHON_BIN=/path/to/python bash build_env_c_plus/build_all.sh`.
