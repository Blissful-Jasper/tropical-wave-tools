# Running the NCL cross-check

Only NCL is needed for the first half; the seeded inputs (`filter_input.nc`, `real_input.nc`)
are committed.

```bash
git clone -b claude/funny-hypatia-27mxnd https://github.com/Blissful-Jasper/tropical-wave-tools.git
cd tropical-wave-tools
bash validation/run_ncl_reference.sh                    # Kelvin only
WAVES=kelvin,er,mrg,mjo,td bash validation/run_ncl_reference.sh   # all bands
```

This writes `validation/reference/ncl_<input>_<wave>.nc`. Then either

- compare locally: `pip install -e . && python validation/compare_with_ncl.py`, or
- commit and push `validation/reference/*.nc`, and the comparison can be run from the repository.

`kf_filter.ncl` is read from `$NCARG_ROOT/lib/ncarg/nclscripts/contrib/` (conda-forge NCL
installs it there). From its header: setting both `hMin` and `hMax` to missing ignores the
dispersion curve (used for `td` and `mjo`); `MRG` and `IG0` are the same branch.
