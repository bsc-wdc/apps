 subroutine getlimits(imin, imax, absmin,absmax, nump, maxp)
 integer imin, imax, absmin,absmax, nump, maxp, d
 d = (absmax-absmin)/maxp
 imin = absmin + nump * (d + 1)
 imax = min (absmax, imin + d)
 end
