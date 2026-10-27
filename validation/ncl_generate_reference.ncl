; Generate NCL kf_filter reference output for validation/compare_with_ncl.py.
;
; Requirements (not shipped here): an NCL install and a copy of the community
; `kf_filter.ncl` (Wheeler-Kiladis filter). Adjust the argument order below if
; your copy differs from kf_filter(x, obsPerDay, tMin, tMax, kMin, kMax, hMin, hMax, waveName).
;
; Run from the repository root after `python validation/make_input.py`:
;     ncl validation/ncl_generate_reference.ncl

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
loadscript("kf_filter.ncl")

f   = addfile("validation/input_equator.nc", "r")
x   = f->olr                    ; (time, lon), anomaly, single latitude
obs = 1                         ; samples per day

; name, tMin, tMax, kMin, kMax, hMin, hMax, NCL waveName
waves = (/ "kelvin", "er", "mrg", "td", "mjo" /)
t_min = (/  3.0,   9.0,  2.5,  2.5, 20.0 /)
t_max = (/ 20.0,  72.0, 10.0,  5.0, 100.0 /)
k_min = (/  2,   -10,   -10,  -20,   1 /)
k_max = (/ 14,    -1,    -1,   -6,   5 /)
h_min = (/  8.0,   8.0,  8.0, -999., -999. /)
h_max = (/ 90.0,  90.0, 90.0, -999., -999. /)
n_name = (/ "Kelvin", "ER", "MRG", "TD", "MJO" /)

do i = 0, dimsizes(waves) - 1
  y = kf_filter(x, obs, t_min(i), t_max(i), k_min(i), k_max(i), h_min(i), h_max(i), n_name(i))
  y!0 = "time"
  y!1 = "lon"
  system("rm -f validation/reference/ncl_" + waves(i) + ".nc")
  g = addfile("validation/reference/ncl_" + waves(i) + ".nc", "c")
  g->filtered = y
end do
