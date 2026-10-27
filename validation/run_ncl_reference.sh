#!/usr/bin/env bash
# One-command NCL reference run for the filter cross-check.
#   1. download kf_filter.ncl from the NCL "kf_filter" documentation page into validation/
#   2. bash validation/run_ncl_reference.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ncl >/dev/null || { echo "ncl not found on PATH"; exit 1; }
[ -f validation/kf_filter.ncl ] || { echo "validation/kf_filter.ncl is missing (download it from the NCL kf_filter page)"; exit 1; }

python validation/make_input.py
mkdir -p validation/reference
ncl validation/ncl_generate_reference.ncl
ls -la validation/reference
python validation/compare_with_ncl.py
