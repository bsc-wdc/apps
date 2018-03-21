!
! Authors : Josep Ll. Gelpi, Agusti Emperador
! Subject : Forcefield module
! Revision: $id$
!
MODULE potentials
 implicit none
!
 integer, parameter:: MAXATYPES = 50

 type atType
   character(len=4) potId
   integer ind
   real qq, gfree, vol, evdw, rvdw, rhc, mas
 end type atType
 
 type ffprm
   integer ntypes 
   type(atType) :: types(MAXATYPES)
 end type ffprm

 character*30, parameter :: potFmt = '(a4,7f8.3)'
 type(atType), parameter :: atTypeNull = atType('',0,0.,0.,0.,0.,0.,0.,0.)

CONTAINS
!=====================================================================================
 function getPotential(ff,idType) result (at)
  type(ffprm), intent(IN) :: ff
  type(atType) at
  character(len=*), intent(IN) ::idType
  integer i
  i=1
  do while (i.lt.ff%ntypes.and.ff%types(i)%potId.ne.idType)
   i=i+1
  enddo
  if (ff%types(i)%potId.eq.idType) then
   at = ff%types(i)
  else
   at = atTypeNull
  endif
 end function getPotential
!=====================================================================================
 subroutine putPotential(ff,at)
  type(ffprm), intent(INOUT) :: ff
  type(atType) at, at1
  
  at1 = getPotential(ff,at%potId)
  if (at1%ind.eq.0) then
   ff%ntypes=ff%ntypes+1
   ff%types(ff%ntypes) = at
  else
   ff%types(at1%ind) = at
   ff%types(at1%ind)%ind = at1%ind
  endif
 end subroutine putPotential
!=====================================================================================  
 function loadPotentials (unt) result (ff)
  type (ffprm) :: ff
  integer unt
  character(len=4) id
  character(len=80) str
  real qq, gfree, vol, evdw, rvdw, rhc, mas

   ff%ntypes=0
10 read (unt,'(a80)', end=20) str
   if (str(1:1).ne.' '.and.str(1:1).ne.'#') then
      read (str, potFmt) id, qq, gfree, vol, evdw, rvdw, rhc, mas
      ff%ntypes = ff%ntypes+1
      ff%types(ff%ntypes) = atType(id, ff%ntypes, qq, gfree, vol, evdw, rvdw, rhc, mas)
   endif
   goto 10
20 continue

 end function loadPotentials
!===================================================================================== 
 subroutine savePotentials (unt,ff)
  type(ffprm) :: ff
  integer unt, i
  write (unt,'(a3)') ff%ntypes
  write (unt,'(1x,a4,": ",a4,7f8.3)') (ff%types(i),i=1,ff%ntypes)
 end subroutine savePotentials

END MODULE potentials
