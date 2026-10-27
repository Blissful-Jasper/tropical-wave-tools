#!/usr/bin/env bash
# NCL half of the filter cross-check. Needs only NCL (the seeded inputs are committed).
#   bash validation/run_ncl_reference.sh                        # Kelvin only
#   WAVES=kelvin,er,mrg,mjo,td,eig,wig bash validation/run_ncl_reference.sh
# Afterwards either run `python validation/compare_with_ncl.py` (needs `pip install -e .`)
# or just commit validation/reference/*.nc and push.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ncl >/dev/null || { echo "ncl not found on PATH"; exit 1; }
kf="${NCARG_ROOT:-}/lib/ncarg/nclscripts/contrib/kf_filter.ncl"
[ -f "$kf" ] || { echo "kf_filter.ncl not found at $kf (is NCARG_ROOT set?)"; exit 1; }

mkdir -p validation/reference
WAVES="${WAVES:-kelvin}" ncl validation/ncl_generate_reference.ncl
ls -la validation/reference
