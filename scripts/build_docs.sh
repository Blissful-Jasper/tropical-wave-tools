#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

MANAGER=""
MODE="build"

print_usage() {
  cat <<'EOF'
Usage: bash scripts/build_docs.sh [options] [build|serve|gh-deploy]

Options:
  --manager <mamba|conda>  Use a specific environment manager.
  -h, --help               Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manager)
      MANAGER="${2:-}"
      shift 2
      ;;
    build|serve|gh-deploy)
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

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
twave_prepare_runtime_dirs
export NO_MKDOCS_2_WARNING=true

echo "[twave-docs] Regenerating gallery assets"
if twave_env_exists "${MANAGER}"; then
  twave_run_in_env "${MANAGER}" python scripts/generate_gallery.py
else
  twave_run_fallback_python scripts/generate_gallery.py
fi

case "${MODE}" in
  build)
    echo "[twave-docs] Building static site"
    if twave_env_exists "${MANAGER}"; then
      twave_run_in_env "${MANAGER}" python -m mkdocs build
    else
      twave_run_fallback_module mkdocs build
    fi
    ;;
  serve)
    echo "[twave-docs] Starting local MkDocs server"
    if twave_env_exists "${MANAGER}"; then
      twave_run_in_env_live "${MANAGER}" python -m mkdocs serve
    else
      twave_run_fallback_module mkdocs serve
    fi
    ;;
  gh-deploy)
    echo "[twave-docs] Deploying to GitHub Pages"
    if ! twave_can_resolve_host "github.com"; then
      echo "[twave-docs] GitHub is not reachable from the current environment." >&2
      exit 2
    fi
    if ! git -C "${PROJECT_ROOT}" remote get-url origin >/dev/null 2>&1; then
      echo "[twave-docs] Git remote 'origin' is not configured." >&2
      exit 2
    fi
    if twave_env_exists "${MANAGER}"; then
      twave_run_in_env "${MANAGER}" python -m mkdocs gh-deploy
    else
      twave_run_fallback_module mkdocs gh-deploy
    fi
    ;;
esac
