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
Usage: bash scripts/run_tests.sh [--manager <mamba|conda>] [pytest args...]
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

cd "${PROJECT_ROOT}"
if twave_env_exists "${MANAGER}"; then
  twave_run_in_env "${MANAGER}" pytest "$@"
else
  twave_run_fallback_python -m pytest "$@"
fi
