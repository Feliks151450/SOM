begin
year = ispan(1948,2014,1)
nyears = dimsizes(year)

f = addfile("E:/huangzy/NECP2/hgt.mon.mean.nc", "r")
time = cd_calendar(f->time, -1)

time_ind = ind(time .ge. 194801 .and. time .le. 201412)
hgt = f->hgt(time_ind,9,{0:90},{0:360})
hgt&time = time(time_ind)
;printVarSummary(hgt)
mean_hgt = dim_avg_n_Wrap(hgt,0)
;mean_hgt = clmMonTLL(hgt)
wks = gsn_open_wks("png", "200hPa mean hgt")

res = True
res@gsnDraw                     = False
res@gsnFrame                    = False ;
res@gsnAddCyclic                = False 
res@cnFillOn                    = False
res@cnLinesOn                   = True
res@cnLineLabelsOn              = True
res@cnInfoLabelOn               = False
res@lbLabelBarOn                = False   
res@lbBoxLinesOn                = False

res@mpMinLatF                   = 0; 左场的空间范围
res@mpMaxLatF                   = 90
; res@mpMinLonF                   = 40
; res@mpMaxLonF                   = 300
; res@mpCenterLonF                = 170
plot = gsn_csm_contour_map(wks,mean_hgt,res)
;print(hgt)
end