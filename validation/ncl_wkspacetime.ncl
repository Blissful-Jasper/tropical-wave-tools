;---------------------------------------------------------------
; Run NCL wkSpaceTime on the seeded cube validation/wk_input.nc
;---------------------------------------------------------------
; Modelled on wk_time4.ncl (Blissful-Jasper/Python_based_Wheeler_Kiladis_Space_Time_Spectra_Analysis),
; whose wkSpaceTime call signature is reused unchanged.
;
; Run from the repository root:   ncl validation/ncl_wkspacetime.ncl
; Afterwards run `ls validation/wk_out*` and tell us which files NCL wrote (PNG only, or also
; netCDF?). The Python side compares the spectra, so we need NCL to save them; if it only
; draws figures, the next step is to add the save-to-netCDF lines for the arrays you name.

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"

latN = 15
latS = -latN
spd  = 1
var  = "olr"

; two segment settings: gap of 30 d between windows (as in wk_time4.ncl) and WK99-style overlap
cases = (/ "wk_skip30", "wk_skipm60" /)
skips = (/ 30, -60 /)

f   = addfile("validation/wk_input.nc", "r")
VAR = f->olr(time|:, {lat|latS:latN}, lon|:)
printVarSummary(VAR)

if (num(ismissing(VAR)) .gt. 0) then
  print("missing values present: not allowed")
  exit
end if

system("mkdir -p validation/wk_out")

do i = 0, dimsizes(cases) - 1
  opt          = True
  opt@debug    = True
  opt@pltType  = "png"
  opt@cnLinesOn = False
  wkSpaceTime(VAR, "validation/wk_out/", cases(i), var, latN, spd, 96, skips(i), opt)
end do
