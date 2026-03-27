#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

MANAGER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manager)
      MANAGER="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_app.sh [--manager <mamba|conda>] [streamlit args...]
EOF
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

MANAGER="$(twave_detect_manager "${MANAGER}")"
twave_prepare_runtime_dirs
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true

cd "${PROJECT_ROOT}"
echo "[twave-app] Starting Streamlit server."
echo "[twave-app] This command stays running until you press Ctrl+C."
echo "[twave-app] Open http://127.0.0.1:8501 after startup, unless you override the port."
if twave_env_exists "${MANAGER}"; then
  twave_run_in_env_live "${MANAGER}" streamlit run apps/streamlit_app.py "$@"
else
  twave_run_fallback_module streamlit run apps/streamlit_app.py "$@"
fi
