#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/environment.yml"
ENV_NAME="$(awk '/^name:/ {print $2; exit}' "${ENV_FILE}")"
export MAMBA_NO_BANNER=1

twave_detect_manager() {
  local requested="${1:-}"
  if [[ -n "${requested}" ]]; then
    if command -v "${requested}" >/dev/null 2>&1; then
      echo "${requested}"
      return
    fi
    echo "Requested manager '${requested}' is not available." >&2
    exit 1
  fi

  if command -v mamba >/dev/null 2>&1; then
    echo "mamba"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return
  fi

  echo "Neither mamba nor conda was found in PATH." >&2
  exit 1
}

twave_env_exists() {
  local manager="$1"
  "${manager}" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"
}

twave_prepare_runtime_dirs() {
  export PYTHONNOUSERSITE=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
}

twave_run_in_env() {
  local manager="$1"
  shift
  "${manager}" run -n "${ENV_NAME}" \
    env \
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}" \
    "$@"
}

twave_run_in_env_live() {
  local manager="$1"
  shift
  "${manager}" run --no-capture-output -n "${ENV_NAME}" \
    env \
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}" \
    "$@"
}

twave_require_env() {
  local manager="$1"
  if ! twave_env_exists "${manager}"; then
    echo "Environment '${ENV_NAME}' does not exist yet." >&2
    echo "Run 'bash scripts/bootstrap_env.sh' first." >&2
    exit 1
  fi
}

twave_can_resolve_host() {
  local host="$1"
  python - <<PY >/dev/null 2>&1
import socket
socket.gethostbyname("${host}")
PY
}

twave_pick_open_port() {
  local preferred="${1:-8000}"
  local host="${2:-127.0.0.1}"
  python - <<PY
import socket

host = "${host}"
preferred = int("${preferred}")
candidates = [preferred, *range(preferred + 1, preferred + 50)]

for port in candidates:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

twave_run_fallback_python() {
  env \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_EXE \
    -u CONDA_PREFIX \
    -u CONDA_PYTHON_EXE \
    -u CONDA_SHLVL \
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    python "$@"
}

twave_run_fallback_module() {
  env \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_EXE \
    -u CONDA_PREFIX \
    -u CONDA_PYTHON_EXE \
    -u CONDA_SHLVL \
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    python "${PROJECT_ROOT}/scripts/run_tool_module.py" "$@"
}
