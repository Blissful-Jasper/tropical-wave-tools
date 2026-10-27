# Running the NCL cross-check

```bash
cd tropical-wave-tools
bash validation/run_ncl_reference.sh              # Kelvin only
```

`kf_filter.ncl` is read from `$NCARG_ROOT/lib/ncarg/nclscripts/contrib/` (conda-forge NCL
installs it there). The call is the same one used in production CCKW scripts: one
`(time, lon)` slice per latitude, lower-case `waveName`.

Before enabling other bands (`WAVES=kelvin,er,mrg,mjo,td`), look at how your `kf_filter.ncl`
treats `waveName` and `hMin`/`hMax`:

```bash
grep -n "waveName\|hMin\|hMax" "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/kf_filter.ncl" | head -60
```

and adjust `h_min`/`h_max` for `td`/`mjo` in `ncl_generate_reference.ncl` if needed.
Paste that grep output into the issue/PR thread if a band disagrees.
