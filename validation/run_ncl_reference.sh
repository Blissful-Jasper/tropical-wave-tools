#!/usr/bin/env bash
# One-command NCL reference run for the filter cross-check.
#   bash validation/run_ncl_reference.sh                        # Kelvin only (confirmed call)
#   WAVES=kelvin,er,mrg,mjo,td bash validation/run_ncl_reference.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ncl >/dev/null || { echo "ncl not found on PATH"; exit 1; }
kf="${NCARG_ROOT:-}/lib/ncarg/nclscripts/contrib/kf_filter.ncl"
[ -f "$kf" ] || { echo "kf_filter.ncl not found at $kf (is NCARG_ROOT set?)"; exit 1; }

python validation/make_input.py
mkdir -p validation/reference
WAVES="${WAVES:-kelvin}" ncl validation/ncl_generate_reference.ncl
ls -la validation/reference
python validation/compare_with_ncl.py
