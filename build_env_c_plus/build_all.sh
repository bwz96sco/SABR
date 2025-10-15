#!/usr/bin/env bash
set -euxo pipefail

# Build native DP binary and pybind11 modules (stop on first error)
bash build_dp.sh
bash build_rl.sh
bash build_rl_mpc.sh
bash build_robustmpc.sh
