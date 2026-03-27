#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

MANAGER=""
RECREATE=0
SKIP_TESTS=0

print_usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_env.sh [options]

Options:
  --manager <mamba|conda>  Use a specific environment manager.
  --recreate               Remove and recreate the environment from scratch.
  --skip-tests             Do not run pytest after installation.
  -h, --help               Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manager)
      MANAGER="${2:-}"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
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

echo "[twave] Project root: ${PROJECT_ROOT}"
echo "[twave] Using ${MANAGER} with environment '${ENV_NAME}'"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if ! twave_env_exists "${MANAGER}" && ! twave_can_resolve_host "conda.anaconda.org"; then
  echo "[twave] Network unavailable. Skipping environment creation and using fallback system Python mode."
  twave_prepare_runtime_dirs
  cd "${PROJECT_ROOT}"
  twave_run_fallback_python -m pytest tests
  echo "[twave] Fallback mode is ready for local test/docs/build commands."
  exit 0
fi

if [[ "${RECREATE}" -eq 1 ]] && twave_env_exists "${MANAGER}"; then
  echo "[twave] Removing existing environment '${ENV_NAME}'"
  "${MANAGER}" env remove -n "${ENV_NAME}" -y
fi

if twave_env_exists "${MANAGER}"; then
  echo "[twave] Updating existing environment from ${ENV_FILE}"
  if ! "${MANAGER}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune; then
    echo "[twave] Environment update failed. Falling back to system Python mode."
    twave_prepare_runtime_dirs
    cd "${PROJECT_ROOT}"
    twave_run_fallback_python -m pytest tests
    echo "[twave] Fallback mode is ready for local test/docs/build commands."
    exit 0
  fi
else
  echo "[twave] Creating environment from ${ENV_FILE}"
  if ! "${MANAGER}" env create -f "${ENV_FILE}"; then
    echo "[twave] Environment creation failed. Falling back to system Python mode."
    twave_prepare_runtime_dirs
    cd "${PROJECT_ROOT}"
    twave_run_fallback_python -m pytest tests
    echo "[twave] Fallback mode is ready for local test/docs/build commands."
    exit 0
  fi
fi

twave_prepare_runtime_dirs

cd "${PROJECT_ROOT}"

echo "[twave] Installing editable package with developer extras"
if ! twave_run_in_env "${MANAGER}" python -m pip install -e ".[dev,docs,app]"; then
  echo "[twave] Editable install failed inside '${ENV_NAME}'."
  exit 1
fi

if [[ "${SKIP_TESTS}" -eq 0 ]]; then
  echo "[twave] Running test suite"
  twave_run_in_env "${MANAGER}" pytest tests
else
  echo "[twave] Skipping tests (--skip-tests)"
fi

echo "[twave] Environment is ready."
echo "[twave] Run commands later with:"
echo "[twave]   ${MANAGER} run -n ${ENV_NAME} python -m mkdocs build"
echo "[twave]   ${MANAGER} run -n ${ENV_NAME} streamlit run apps/streamlit_app.py"
