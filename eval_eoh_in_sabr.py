#!/usr/bin/env python3
"""Bridge script: run an EoH-evolved heuristic through SABR's fixed_env
simulation and write per-trace log files in SABR's format.

Usage:
    python eval_eoh_in_sabr.py --json path/to/population.json [--index 0] [--dataset FCC-18]
    python eval_eoh_in_sabr.py --py   path/to/heuristic.py   [--dataset FCC-18]
    python eval_eoh_in_sabr.py --code 'def score(state, ctx): ...' [--dataset FCC-18]
    python eval_eoh_in_sabr.py --json path/to/population.json --dataset FCC-18 --log-dir path/to/logs

Output:
    Per-trace log files in the dataset's LOG_FILE_DIR, or the explicit --log-dir:
        log_sim_eoh_<trace_name>
    Each line: time_stamp\tbitrate\tbuffer\trebuf\tchunk_size\tdelay\treward
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths so imports work regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent                    # env/SABR/
REPO_ROOT = SCRIPT_DIR.parents[1]                               # EoH/
ABR_EXAMPLE_DIR = REPO_ROOT / "examples" / "user_abr"

# Make SABR root importable (for config, sim_env, etc.)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# Make user_abr importable (for abr_api helpers)
if str(ABR_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(ABR_EXAMPLE_DIR))

from abr_api import extract_future_chunk_sizes, extract_state, make_ctx  # noqa: E402
from config import _DATASET_OPTION                                       # noqa: E402
import sim_env.fixed_env as env                                          # noqa: E402
from sim_env import load_trace                                           # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match bb.py / prob.py)
# ---------------------------------------------------------------------------
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
HISTORY_WINDOW = 10
MPC_FUTURE_CHUNK_COUNT = 5
SMOOTH_PENALTY = 1
M_IN_K = 1000.0
EPS = 1e-6
SCHEME_NAME = "eoh"


# ---------------------------------------------------------------------------
# Heuristic loading
# ---------------------------------------------------------------------------

def load_score_fn_from_json(path: str, index: int = 0):
    """Load a score(state, ctx) function from EoH population JSON.

    Handles two formats:
    - List of dicts: population[index]["code"]
    - Single dict (pops_best): population["code"]
    """
    with open(path) as f:
        population = json.load(f)
    if isinstance(population, dict) and "code" in population:
        entry = population
    elif isinstance(population, list):
        entry = population[index]
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")
    code = entry["code"]
    return _compile_score(code), code


def load_score_fn_from_py(path: str):
    """Load a score(state, ctx) function from a .py file."""
    code = Path(path).read_text()
    return _compile_score(code), code


def load_score_fn_from_code(code: str):
    """Load a score(state, ctx) function from raw code string."""
    return _compile_score(code), code


def _compile_score(code: str):
    mod = types.ModuleType("eoh_heuristic")
    exec(code, mod.__dict__)
    fn = getattr(mod, "score", None)
    if fn is None:
        raise ValueError("Heuristic code does not define a `score(state, ctx)` function.")
    return fn


# ---------------------------------------------------------------------------
# Simulation + log writing (mirrors bb.py loop structure)
# ---------------------------------------------------------------------------

def run_evaluation(
    score_fn,
    dataset: str,
    label: str = SCHEME_NAME,
    log_dir: str | None = None,
) -> float:
    """Run score_fn on every trace in *dataset* and write SABR-format logs.

    Returns average per-video reward.
    """
    ds = _DATASET_OPTION[dataset]
    video_bit_rate = ds["VIDEO_BIT_RATE"]
    rebuf_penalty = ds["REBUF_PENALTY"]
    test_traces = ds["TEST_TRACES"]
    log_file_dir = log_dir or ds["LOG_FILE_DIR"]

    os.makedirs(log_file_dir, exist_ok=True)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)

    np.random.seed(RANDOM_SEED)

    net_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
    )

    # Build ctx dict (same as prob.py)
    import importlib.util
    env_path = SCRIPT_DIR / "sim_env" / "fixed_env.py"
    spec = importlib.util.spec_from_file_location("_env_consts", env_path)
    env_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_mod)

    # Construct a minimal config-like object for make_ctx
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.VIDEO_BIT_RATE = video_bit_rate
    cfg.REBUF_PENALTY = rebuf_penalty
    ctx = make_ctx(cfg, env=env_mod, smooth_penalty=SMOOTH_PENALTY, rebuf_penalty=rebuf_penalty)

    video_bit_rate_arr = np.asarray(video_bit_rate, dtype=np.float64)
    max_bitrate_idx = len(video_bit_rate) - 1

    log_prefix = os.path.join(log_file_dir, f"log_sim_{label}")
    log_path = log_prefix + "_" + all_file_names[net_env.trace_idx]
    log_file = open(log_path, "w")

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    throughput_history = deque(maxlen=HISTORY_WINDOW)

    r_batch = []
    batch_rewards = []
    video_count = 0

    while True:
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay
        time_stamp += sleep_time

        # Reward (same formula as bb.py / prob.py)
        reward = (
            video_bit_rate[bit_rate] / M_IN_K
            - rebuf_penalty * rebuf
            - SMOOTH_PENALTY * abs(video_bit_rate[bit_rate] - video_bit_rate[last_bit_rate]) / M_IN_K
        )
        r_batch.append(reward)

        last_bit_rate = bit_rate

        # Write log line (identical format to bb.py)
        log_file.write(
            f"{time_stamp / M_IN_K}\t"
            f"{video_bit_rate[bit_rate]}\t"
            f"{buffer_size}\t"
            f"{rebuf}\t"
            f"{video_chunk_size}\t"
            f"{delay}\t"
            f"{reward}\n"
        )
        log_file.flush()

        # --- Compute throughput and build state for heuristic ---
        if delay > EPS:
            throughput_kbps = (float(video_chunk_size) * 8.0) / float(delay)
        else:
            throughput_kbps = float(video_bit_rate[bit_rate])
        throughput_history.append(throughput_kbps)

        # Build future chunk sizes matrix
        future_chunk_sizes_bytes = extract_future_chunk_sizes(
            net_env,
            video_chunk_remain,
            horizon=MPC_FUTURE_CHUNK_COUNT,
        )

        state = extract_state(
            obs=None,
            info=(
                delay,
                sleep_time,
                buffer_size,
                rebuf,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain,
            ),
            last_action=last_bit_rate,
            throughput_history=np.asarray(throughput_history, dtype=np.float64),
            future_chunk_sizes_bytes=future_chunk_sizes_bytes,
        )

        # --- Call heuristic ---
        try:
            scores = np.asarray(score_fn(state, ctx), dtype=np.float64).reshape(-1)
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            if scores.size == len(video_bit_rate) and np.isfinite(scores).any():
                bit_rate = int(np.argmax(scores))
            else:
                bit_rate = last_bit_rate
        except Exception:
            bit_rate = last_bit_rate

        bit_rate = int(np.clip(bit_rate, 0, max_bitrate_idx))

        if end_of_video:
            log_file.write("\n")
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            throughput_history.clear()

            batch_reward = np.sum(r_batch)
            batch_rewards.append(batch_reward)
            r_batch = []

            print(f"video count {video_count}")
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = log_prefix + "_" + all_file_names[net_env.trace_idx]
            log_file = open(log_path, "w")
            time_stamp = 0

    total_reward = np.sum(batch_rewards)
    average_reward = total_reward / len(batch_rewards) if batch_rewards else 0.0
    print(f"total reward {total_reward}")
    print(f"average reward {average_reward}")
    print(f"Logs written to {log_file_dir}")
    return average_reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an EoH heuristic via SABR sim")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="Path to EoH population JSON file")
    src.add_argument("--py", help="Path to a .py file defining score(state, ctx)")
    src.add_argument("--code", help="Raw Python code string defining score(state, ctx)")
    p.add_argument("--index", type=int, default=0,
                   help="Index into population JSON (default: 0)")
    p.add_argument("--dataset", default="FCC-18",
                   choices=list(_DATASET_OPTION.keys()),
                   help="Dataset to evaluate on (default: FCC-18)")
    p.add_argument("--label", default=SCHEME_NAME,
                   help="Scheme label for log filenames (default: eoh)")
    p.add_argument(
        "--log-dir",
        default=None,
        help="Optional output directory for SABR-format logs. Defaults to the dataset LOG_FILE_DIR.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.json:
        score_fn, code = load_score_fn_from_json(args.json, args.index)
        print(f"Loaded heuristic from JSON: {args.json} [index={args.index}]")
    elif args.py:
        score_fn, code = load_score_fn_from_py(args.py)
        print(f"Loaded heuristic from .py: {args.py}")
    else:
        score_fn, code = load_score_fn_from_code(args.code)
        print("Loaded heuristic from --code argument")

    print(f"Dataset: {args.dataset}")
    print(f"Label: {args.label}")
    if args.log_dir:
        print(f"Log dir override: {args.log_dir}")
    print("---")

    run_evaluation(
        score_fn,
        dataset=args.dataset,
        label=args.label,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
