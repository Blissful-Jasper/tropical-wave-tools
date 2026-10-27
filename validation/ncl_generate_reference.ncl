; Generate NCL kf_filter golden files for validation/compare_with_ncl.py.
;
; >>> UNVERIFIED DRAFT <<<
; The kf_filter call below (argument order, "unused" sentinel values, waveName strings)
; was written from memory, not from the NCL documentation. Check it against
;   https://www.ncl.ucar.edu/Document/Functions/User_contributed/kf_filter.shtml
; and your local kf_filter.ncl before trusting any comparison result.
;
; Run from the repository root after `python validation/make_input.py`:
;     ncl validation/ncl_generate_reference.ncl
; Output: validation/reference/ncl_<input>_<wave>.nc  (variable "filtered", dims time/lat/lon)

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
loadscript("validation/kf_filter.ncl")   ; download from the NCL kf_filter doc page, not committed

inputs = (/ "filter", "real" /)
obs    = 1                         ; samples per day

; python wave name, tMin, tMax, kMin, kMax, hMin, hMax, NCL waveName
waves  = (/ "kelvin", "er", "mrg", "td", "mjo" /)
t_min  = (/  3.0,   9.0,  2.5,  2.5,  20.0 /)
t_max  = (/ 20.0,  72.0, 10.0,  5.0, 100.0 /)
k_min  = (/  2,   -10,   -10,  -20,    1 /)
k_max  = (/ 14,    -1,    -1,   -6,    5 /)
h_min  = (/  8.0,   8.0,  8.0, -999., -999. /)
h_max  = (/ 90.0,  90.0, 90.0, -999., -999. /)
n_name = (/ "Kelvin", "ER", "MRG", "none", "none" /)   ; TODO verify names for TD / MJO

do s = 0, dimsizes(inputs) - 1
  f = addfile("validation/" + inputs(s) + "_input.nc", "r")
  x = f->x                                    ; (time, lat, lon)
  do i = 0, dimsizes(waves) - 1
    y = x
    do j = 0, dimsizes(x&lat) - 1             ; filter one latitude row at a time
      y(:, j, :) = (/ kf_filter(x(:, j, :), obs, t_min(i), t_max(i), \
                                k_min(i), k_max(i), h_min(i), h_max(i), n_name(i)) /)
    end do
    path = "validation/reference/ncl_" + inputs(s) + "_" + waves(i) + ".nc"
    system("rm -f " + path)
    g = addfile(path, "c")
    g->filtered = y
  end do
end do
