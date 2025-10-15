#!/usr/bin/env bash
set -euo pipefail

# Prefer project venv Python if present; fallback to system python3
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "../.venv/bin/python" ]]; then
    PYTHON_BIN="../.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

INCLUDES="$($PYTHON_BIN -m pybind11 --includes)"
PLATFORM_LDFLAGS=""
if [[ "$(uname)" == "Darwin" ]]; then
  PLATFORM_LDFLAGS="-undefined dynamic_lookup"
fi
EXT_SUFFIX="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_config_var('EXT_SUFFIX') or sysconfig.get_config_var('SO'))
PY
)"

c++ -O3 -Wall -shared -std=c++11 -fPIC ${INCLUDES} ${PLATFORM_LDFLAGS} corerl.cc -o "libcorerl${EXT_SUFFIX}"
cp libcorerl*.so ../sim_env
echo 'Done'
