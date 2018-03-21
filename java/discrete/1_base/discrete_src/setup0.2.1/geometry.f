!
! $Date: 2011-09-01 18:15:51 $
! $Id: geometry.f,v 1.2 2011-09-01 18:15:51 gelpi Exp $
! $Revision: 1.2 $
!
MODULE geometry

TYPE point 
  REAL :: x,y,z 
END TYPE point

TYPE pointPolar
  REAL :: r,phi,the
END TYPE pointPolar

TYPE pointInt
  integer :: i,j,k
END TYPE pointInt

real, parameter :: PI=3.1415926

interface operator (+) 
module procedure sumavec
end interface

interface operator (-)
module procedure negvec, restavec, restaint
end interface

interface operator (*)
module procedure prodFact, dot
end interface

interface assignment (=)
module procedure topint,topreal
end interface

CONTAINS

pure function sumavec (v1,v2)
type (point):: sumavec
TYPE (point), intent (IN):: v1,v2
sumavec%x=v1%x+v2%x
sumavec%y=v1%y+v2%y
sumavec%z=v1%z+v2%z
end function sumavec

pure function prodFact (f,v)
type (point) :: prodFact
TYPE (point), intent (IN):: v
real, intent (IN):: f
prodFact%x=v%x*f
prodFact%y=v%y*f
prodFact%z=v%z*f
end function prodFact

pure function negvec (v)
type (point) :: negvec
type (point), intent (in) :: v
negvec = prodFact (-1.0, v)
end function negvec

pure function restaVec (v1,v2)
type (point) :: restaVec
type (point), intent(IN) :: v1, v2
restaVec = v1 + (-v2)
end function restaVec

pure function dot (v1,v2)
type (point), intent (IN):: v1,v2
real :: dot
dot=v1%x*v2%x+v1%y*v2%y+v1%z*v2%z
end function dot

pure function cross(v1,v2)
type (point) cross
type (point), intent (IN) :: v1,v2
cross=point(-v1%z*v2%y + v1%y*v2%z, v1%z*v2%x - v1%x*v2%z, -v1%y*v2%x + v1%x*v2%y)
end function cross

pure function module (v)
real :: module
type (point), intent (IN) :: v
module = sqrt (v*v)
end function module

pure function makeUnit (v)
type(point):: makeUnit
type (point), intent (IN) :: v
makeUnit = (1/module(v))*v
end function makeUnit

pure function cosang (v1,v2)
real :: cosang
type (point), intent (IN) :: v1,v2
cosang=(v1*v2)/module(v1)/module(v2)
end function

subroutine topint (a,p)
type (pointInt), intent(OUT) :: a
type (point), intent (IN) :: p
a%i=int(p%x)
a%j=int(p%y)
a%k=int(p%z)
end subroutine topint

subroutine topreal (a,p)
type (point), intent (OUT) :: a
type (pointInt), intent (IN) :: p
a%x=p%i
a%y=p%j
a%z=p%k
end subroutine topreal

pure function restaint (p,ip)
type (point) restaint
type (point), intent (IN) :: p
type (pointInt), intent (IN) :: ip
restaint%x=p%x-ip%i
restaint%y=p%y-ip%j
restaint%z=p%z-ip%k
end function restaint

pure FUNCTION calcDist2 (p1,p2)
  IMPLICIT NONE
  REAL :: calcDist2
  TYPE (point), INTENT(IN) :: p1,p2
  type (point) :: v
  v = restaVec(p1,p2)
  calcDist2 = v * v
END FUNCTION calcDist2

pure FUNCTION calcDist (p1,p2)
  IMPLICIT NONE
  REAL :: calcDist
  TYPE (point), INTENT(IN) :: p1,p2
  calcDist = SQRT (calcDist2 (p1,p2))
END FUNCTION calcDist

pure function rotx (p,cint,sint)
implicit none
type (point), intent (in) :: p
type (point) :: rotx
real, intent (IN) :: cint, sint
rotx%x=p%x
rotx%y=p%y*cint-p%z*sint
rotx%z=p%y*sint+p%z*cint
end function rotx

pure function roty (p,cint,sint)
type (point) :: roty
type (point), intent (IN) :: p
real, intent (IN) :: cint,sint
roty%y=p%y
roty%x=p%x*cint-p%z*sint
roty%z=p%x*sint+p%z*cint
end function roty

pure function rotz (p,cint,sint)
type (point) :: rotz
type (point), intent (IN) :: p
real, intent (IN) :: cint,sint
rotz%x=p%x*cint-p%y*sint
rotz%y=p%x*sint+p%y*cint
rotz%z=p%z
end function rotz

pure function rotphi (p,cint,sint)
type (point) :: rotphi
type (point), intent (IN) :: p
real, intent (IN) :: cint,sint
rotphi= rotz (p,cint,sint)
end function rotphi

pure function rotthe (p,cint,sint)
type (point) :: rotthe
type (point), intent (IN) :: p
real, intent (IN) :: cint,sint
real :: rxy
rxy=sqrt(p%x**2+p%y**2)
if (rxy.ne.0) then
  rotthe%x=p%x*cint+p%z*p%x/rxy*sint
else
  rotthe%x=p%x*cint
end if
if (rxy.ne.0) then
  rotthe%y=p%y*cint+p%z*p%y/rxy*sint
else
  rotthe%y=p%y*cint
end if
rotthe%z=p%z*cint-rxy*sint
end function rotthe

pure function rota (p, func, cint, sint)
type (point) :: rota
type (point), intent (IN) :: p
character (len=3), intent (IN) :: func
real, intent(IN) :: cint, sint
select case (func)
 case ('x')
   rota = rotx(p,cint,sint)
 case ('y')
   rota = roty(p,cint,sint)
 case ('z')
   rota = rotz(p,cint,sint)
 case ('phi')
   rota = rotphi(p,cint,sint)
 case ('the')
   rota = rotthe(p,cint,sint)
end select
end function rota

pure function rotaAng (p,func,a)
type (point) :: rotaAng
type (point), intent (IN) ::p
character(len=3), intent (IN) :: func
real, intent(IN) :: a
real cint, sint
cint = cos (a)
sint = sin (a)
rotaAng = rota (p,func,cint,sint)
end function rotaAng

pure function pol2car (p)
type (point):: pol2car
type (pointPolar), intent(IN):: p
pol2car%x=p%r*sin(p%the)*cos(p%phi)
pol2car%y=p%r*sin(p%the)*sin(p%phi)
pol2car%z=p%r*cos(p%the)
end function pol2car

pure function car2pol (p)
type (point), intent (IN):: p
type (pointPolar):: car2pol
car2pol%r=module(p)
  car2pol%phi=atan2(p%y,p%x)
 car2pol%the=atan2(sqrt(p%x**2+p%y**2),p%z)
end function car2pol

pure function dist2pol (p1,p2)
type (pointPolar), intent(IN) :: p1,p2
real :: dist2pol
dist2pol= calcDist2 (pol2car(p1),pol2car(p2))
end function dist2pol

pure function rotquat (p1,pv,sint,cint)
type (point), intent (IN) :: p1,pv
type (point) :: rotquat
real, intent (IN) :: sint,cint
real :: w,x,y,z
x=pv%x*sint
y=pv%y*sint
z=pv%z*sint
w=cint
rotquat%x=p1%x*(1-2*(y**2+z**2))+p1%y*2*(x*y+w*z)+p1%z*2*(x*z+w*y)
rotquat%y=p1%x*2*(x*y+w*z)+p1%y*(1-2*(x**2+z**2))+p1%z*2*(y*z-w*x)
rotquat%z=p1%x*2*(x*z-w*y)+p1%y*2*(y*z+w*x)+p1%z*(1-2*(x**2+y**2))
end function rotquat

pure function rotEuler (p1,s1,c1,s2,c2,s3,c3)
type (point), intent (IN) :: p1
type (point) :: rotEuler
real, intent(IN) ::  s1,c1,s2,c2,s3,c3
rotEuler%x = p1%x*c3*c2*c1 -p1%x*s3*s1 +p1%y*c3*c2*s1 +p1%y*s3*c1 -p1%z*c3*s2
rotEuler%y =-p1%x*s3*c2*c1 -p1%x*c3*s1 -p1%y*s3*c2*s1 +p1%y*c3*c1 +p1%z*s3*s2
rotEuler%z = p1%x*s2*c1                +p1%y*s2 *s1              +p1%z*c2
end function rotEuler

function rotEuler1 (s1,c1,s2,c2,s3,c3)
type (point) :: rotEuler1
real, intent(IN):: s1,c1,s2,c2,s3,c3
rotEuler1 = rotEuler(point(1.0,0.0,0.0),s1,c1,s2,c2,s3,c3)
end function rotEuler1

function rotaEix (r,n0,s1,c1) 
type (point) rotaEix
type (point), intent (IN) :: r,n0
type (point) n
real, intent (IN) :: s1,c1
n=makeUnit(n0)
rotaEix=(c1 * r + s1 * cross(r,n)) + (dot(n,r)*(1 - c1) * n)

end function rotaEix


END MODULE geometry
