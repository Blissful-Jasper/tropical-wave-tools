; Generate NCL kf_filter golden files for validation/compare_with_ncl.py.
;
; The kf_filter call mirrors a working production script (Xianpu Ji, Dec 2024):
;   load ".../nclscripts/contrib/kf_filter.ncl"      ; ships with conda-forge NCL
;   filtered(time|:, lat|y, lon|:) = kf_filter(x(time|:, lat|y, lon|:), obsPerDay,
;                 period(0), period(1), wavenumber(0), wavenumber(1), depth(0), depth(1), "kelvin")
; i.e. one (time, lon) slice per latitude and a lower-case waveName.
;
; kf_filter.ncl (header, lines 14-24): setting BOTH hMin and hMax to *missing* ignores the
; dispersion curve, which is what the rectangular "mjo" and "td" bands need. Accepted waveName
; values are Kelvin/kelvin/KELVIN, ER/er, MRG/IG0/mrg/ig0, IG1/ig1, IG2/ig2; any other name only
; applies the period/wavenumber box. Band edges follow tropical_wave_tools.config.DEFAULT_WAVE_SPECS.
;
; Run from the repository root (see validation/run_ncl_reference.sh):
;     WAVES=kelvin ncl validation/ncl_generate_reference.ncl
;     WAVES=kelvin,er,mrg,mjo,td ncl validation/ncl_generate_reference.ncl

load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/kf_filter.ncl"

begin
  wave_env = getenv("WAVES")
  if (ismissing(wave_env)) then
    wave_env = "kelvin"
  end if
  waves  = str_split(wave_env, ",")
  inputs = (/ "filter", "real" /)
  obsPerDay = 1

  ; name        tMin  tMax  kMin kMax
  names = (/ "kelvin", "er", "mrg", "td", "mjo" /)
  t_min = (/  3,    9,  2.5,  2.5,  20 /)
  t_max = (/ 20,   72,   10,    5, 100 /)
  k_min = (/  2,  -10,  -10,  -20,   1 /)
  k_max = (/ 14,   -1,   -1,   -6,   5 /)

  ; equivalent depths; entries left unset stay missing = "no dispersion curve" (td, mjo)
  h_min = new(5, float)
  h_max = new(5, float)
  h_min(0:2) = (/ 8,  8,  8 /)
  h_max(0:2) = (/ 90, 90, 90 /)

  do s = 0, dimsizes(inputs) - 1
    f = addfile("validation/" + inputs(s) + "_input.nc", "r")
    x = f->x                                   ; (time, lat, lon)
    nlat = dimsizes(x&lat)

    do w = 0, dimsizes(waves) - 1
      i = ind(names .eq. waves(w))
      if (ismissing(i)) then
        print("unknown wave: " + waves(w))
        exit
      end if
      y = new(dimsizes(x), float)
      y!0 = "time"
      y!1 = "lat"
      y!2 = "lon"
      do j = 0, nlat - 1
        y(time|:, lat|j, lon|:) = kf_filter(x(time|:, lat|j, lon|:), obsPerDay,   \
                                  t_min(i), t_max(i), k_min(i), k_max(i),          \
                                  h_min(i), h_max(i), waves(w))
      end do
      path = "validation/reference/ncl_" + inputs(s) + "_" + waves(w) + ".nc"
      system("rm -f " + path)
      g = addfile(path, "c")
      g->filtered = y
      delete(g)
      delete(y)
      delete(i)
      print("wrote " + path)
    end do
    delete(x)
  end do
end
