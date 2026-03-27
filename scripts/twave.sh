#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
  cat <<'EOF'
Usage: bash scripts/twave.sh <command> [options]

Commands:
  setup        Create/update the development environment.
  test         Run pytest inside the project environment.
  docs         Build documentation.
  docs-serve   Serve documentation locally.
  pages        Deploy MkDocs to GitHub Pages.
  app          Run the Streamlit app.
  build        Build wheel and sdist.
  publish      Build and upload to PyPI.
  publish-test Build and upload to TestPyPI.
EOF
}

if [[ $# -eq 0 ]]; then
  print_usage
  exit 0
fi

COMMAND="$1"
shift

case "${COMMAND}" in
  setup)
    exec bash "${SCRIPT_DIR}/bootstrap_env.sh" "$@"
    ;;
  test)
    exec bash "${SCRIPT_DIR}/run_tests.sh" "$@"
    ;;
  docs)
    exec bash "${SCRIPT_DIR}/build_docs.sh" build "$@"
    ;;
  docs-serve)
    exec bash "${SCRIPT_DIR}/build_docs.sh" serve "$@"
    ;;
  pages)
    exec bash "${SCRIPT_DIR}/build_docs.sh" gh-deploy "$@"
    ;;
  app)
    exec bash "${SCRIPT_DIR}/run_app.sh" "$@"
    ;;
  build)
    exec bash "${SCRIPT_DIR}/release_pypi.sh" build "$@"
    ;;
  publish)
    exec bash "${SCRIPT_DIR}/release_pypi.sh" publish "$@"
    ;;
  publish-test)
    exec bash "${SCRIPT_DIR}/release_pypi.sh" --testpypi publish "$@"
    ;;
  -h|--help|help)
    print_usage
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    print_usage >&2
    exit 1
    ;;
esac

