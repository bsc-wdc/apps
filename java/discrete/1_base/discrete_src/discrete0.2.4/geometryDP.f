!
! $Date: 2011-08-30 10:54:35 $
! $Id: geometryDP.f,v 1.1 2011-08-30 10:54:35 gelpi Exp $
! $Revision: 1.1 $
!
 MODULE geometryDP

 TYPE pointDP 
   REAL*8 :: x,y,z 
 END TYPE pointDP

 interface operator (+) 
  module procedure sumavecDP
 end interface

 interface operator (-)
  module procedure negvecDP, restavecDP
 end interface

 interface operator (*)
  module procedure prodFactDP, dotDP
 end interface

CONTAINS

pure function SPtoDP (rsp) result (rdp)
use geometry
 type(point), intent(IN) :: rsp
 type(pointDP) rdp
 rdp%x=rsp%x
 rdp%y=rsp%y
 rdp%z=rsp%z
end function SPtoDP

pure function DPtoSP (rdp) result (rsp)
use geometry
 type(point) rsp
 type(pointDP), intent(IN) :: rdp
 rsp%x=rdp%x
 rsp%y=rdp%y
 rsp%z=rdp%z
end function DPtoSP

pure function sumavecDP (v1,v2)
type (pointDP):: sumavecDP
TYPE (pointDP), intent (IN):: v1,v2
sumavecDP%x=v1%x+v2%x
sumavecDP%y=v1%y+v2%y
sumavecDP%z=v1%z+v2%z
end function sumavecDP

pure function prodFactDP (f,v)
type (pointDP) :: prodFactDP
TYPE (pointDP), intent (IN):: v
real, intent (IN):: f
prodFactDP%x=v%x*f
prodFactDP%y=v%y*f
prodFactDP%z=v%z*f
end function prodFactDP

pure function negvecDP (v)
type (pointDP) :: negvecDP
type (pointDP), intent (in) :: v
negvecDP = prodFactDP (-1.0, v)
end function negvecDP

pure function restaVecDP (v1,v2)
type (pointDP) :: restaVecDP
type (pointDP), intent(IN) :: v1, v2
restaVecDP = v1 + (-v2)
end function restaVecDP

pure function dotDP (v1,v2)
type (pointDP), intent (IN):: v1,v2
real :: dotDP
dotDP=v1%x*v2%x+v1%y*v2%y+v1%z*v2%z
end function dotDP

pure function dotDP2 (v1,v2)
type (pointDP), intent (IN):: v1,v2
real*8 :: dotDP2
dotDP2=v1%x*v2%x+v1%y*v2%y+v1%z*v2%z
end function dotDP2

pure function crossDP(v1,v2)
type (pointDP) crossDP
type (pointDP), intent (IN) :: v1,v2
crossDP=pointDP(-v1%z*v2%y + v1%y*v2%z, v1%z*v2%x - v1%x*v2%z, -v1%y*v2%x + v1%x*v2%y)
end function crossDP

pure function moduleDP (v)
real :: moduleDP
type (pointDP), intent (IN) :: v
moduleDP = sqrt (v*v)
end function moduleDP

pure function makeUnitDP (v)
type(pointDP):: makeUnitDP
type (pointDP), intent (IN) :: v
makeUnitDP = (1./moduleDP(v))*v
end function makeUnitDP

pure function cosangDP (v1,v2)
real :: cosangDP
type (pointDP), intent (IN) :: v1,v2
cosangDP=(v1*v2)/moduleDP(v1)/moduleDP(v2)
end function cosangDP

pure FUNCTION calcDist2DP (p1,p2)
  IMPLICIT NONE
  REAL :: calcDist2DP
  TYPE (pointDP), INTENT(IN) :: p1,p2
  type (pointDP) :: v
  v = restaVecDP(p1,p2)
  calcDist2DP = v * v
END FUNCTION calcDist2DP

pure FUNCTION calcDistDP (p1,p2)
  IMPLICIT NONE
  REAL :: calcDistDP
  TYPE (pointDP), INTENT(IN) :: p1,p2
  calcDistDP = SQRT (calcDist2DP (p1,p2))
END FUNCTION calcDistDP

END MODULE geometryDP
