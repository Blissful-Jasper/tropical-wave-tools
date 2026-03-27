#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

MANAGER=""
MODE="build"
USE_TESTPYPI=0

print_usage() {
  cat <<'EOF'
Usage: bash scripts/release_pypi.sh [options] [build|publish]

Options:
  --manager <mamba|conda>  Use a specific environment manager.
  --testpypi               Upload to TestPyPI instead of PyPI.
  -h, --help               Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manager)
      MANAGER="${2:-}"
      shift 2
      ;;
    --testpypi)
      USE_TESTPYPI=1
      shift
      ;;
    build|publish)
      MODE="$1"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

MANAGER="$(twave_detect_manager "${MANAGER}")"
twave_prepare_runtime_dirs

cd "${PROJECT_ROOT}"

if [[ "${MODE}" == "publish" ]]; then
  if [[ "${USE_TESTPYPI}" -eq 1 ]]; then
    if ! twave_can_resolve_host "test.pypi.org"; then
      echo "[twave-release] TestPyPI is not reachable from the current environment." >&2
      exit 2
    fi
  else
    if ! twave_can_resolve_host "pypi.org"; then
      echo "[twave-release] PyPI is not reachable from the current environment." >&2
      exit 2
    fi
  fi
fi

echo "[twave-release] Building package"
if twave_env_exists "${MANAGER}"; then
  twave_run_in_env "${MANAGER}" python -m build --no-isolation
else
  twave_run_fallback_module build --no-isolation
fi

if [[ "${MODE}" == "publish" ]]; then
  if [[ "${USE_TESTPYPI}" -eq 1 ]]; then
    echo "[twave-release] Uploading to TestPyPI"
  else
    echo "[twave-release] Uploading to PyPI"
  fi
  if [[ "${USE_TESTPYPI}" -eq 1 ]]; then
    if twave_env_exists "${MANAGER}"; then
      twave_run_in_env "${MANAGER}" twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    else
      twave_run_fallback_module twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    fi
  else
    if twave_env_exists "${MANAGER}"; then
      twave_run_in_env "${MANAGER}" twine upload dist/*
    else
      twave_run_fallback_module twine upload dist/*
    fi
  fi
fi
