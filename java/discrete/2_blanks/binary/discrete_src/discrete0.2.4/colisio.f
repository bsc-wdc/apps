 subroutine colisioBond(bpairs, blist, r, v, rb, drb, xm, xsum, natom)
 use geometry
 use geometryDP  
   integer, intent(IN) :: natom, bpairs
   integer, intent(IN) :: blist(bpairs,2)
   type(pointDP), intent(IN) :: r(natom)
   real, intent(IN) :: rb(natom,natom), drb(natom,natom), xm(natom), xsum(natom,natom)
   type(point), intent(INOUT) :: v(natom)

   real csang, calcSignCosAng
   real rbmin, rbmax, rij
   integer i,j, np
   do np = 1,bpairs
      i = blist(np,1)
      j = blist(np,2)
      csang = calcSignCosAng(r, v, i, j, natom)
      rij = calcDistDP(r(i), r(j))
      rbmin = rb(i,j) - drb(i,j)
      rbmax = rb(i,j) + drb(i,j)
      if((rij.gt.rbmax.and.csang.gt.0.).or.(rij.lt.rbmin.and.csang.lt.0.))  then
         call chgmom(i, j, xm, xsum(i,j), r, v, natom)
      endif
   enddo
   return
 end subroutine colisioBond
!===============================================================================
 subroutine colisioNonBond(stepPts, temps, nblist, r, v, rhc, shc, ind2, atom, xm, ierr, &
               TMIN, natom, isolv)
 use geometry
 use geometryDP
 use intList
   integer, intent(IN) :: natom, isolv
   type(intPList), intent(INOUT) :: nblist(natom)
   integer, intent(IN) :: ind2(natom)
   type(pointDP), intent(IN) :: r(natom)
   type(stepPotInt), intent(INOUT) :: stepPts(natom,natom)
   real, intent(IN) :: rhc(natom), xm(natom), TMIN, shc
   character(len=4),intent(IN) :: atom(natom)

   integer, intent(INOUT) :: ierr
   type(point), intent(INOUT) :: v(natom)
   
   real csang, dmin, rij, calcSignCosAng
   integer i, j, k, l
   logical isalt
   real*8 temps
!
! removes overlaps
   do i = 1, natom
   do k = 1, nblist(i)%nats
      j = nblist(i)%iData(k)%patnum
      if (j.gt.i) then
         csang = calcSignCosAng(r, v, i, j, natom)
         dmin = (rhc(i) + rhc(j) - shc)**2
! evita els xocs dels molt adjacents
         isalt = (ind2(j).eq.ind2(i)+1).and.( &
                 (atom(i).eq.'C'.and.atom(j).eq.'CB').or. &
                 (atom(i).eq.'CB'.and.atom(j).eq.'N').or. &
                 (atom(i).eq.'O'.and.atom(j).eq.'CA'))
         if (.not.isalt) then
            rij = calcDist2DP(r(i), r(j))
            if (rij.lt.dmin.and.csang.lt.0) then
               call chgmom(i, j, xm, nblist(i)%iData(k)%xsum, r, v, natom)
               ierr=ierr+1
            endif
         endif
      endif
   enddo
   enddo
! calculate colision times
   do i = 1, natom
   do k = 1, nblist(i)%nats
      j = nblist(i)%iData(k)%patnum
      l = nblist(i)%iData(k)%simp
      if (j.gt.i) then
         if(stepPts(i,j)%tipInt.eq.SS.or.ISOLV.eq.1) then
            call updateTCol (i, temps, r, v, nblist(i)%iData(k), TMIN, natom)
            nblist(j)%iData(l)%timp   = nblist(i)%iData(k)%timp
            nblist(j)%iData(l)%deltak = nblist(i)%iData(k)%deltak
         endif
      endif
   enddo
   enddo
 end subroutine colisioNonBond
!===============================================================================
 pure subroutine chgmom(mem1, mem2, xm, xsum, r, v, natom)
 use geometry
 use geometryDP
   integer, intent(IN) :: natom
   real, intent(IN) :: xsum, xm(natom)
   type(pointDP), intent(IN) :: r(natom)
   integer, intent(IN) :: mem1, mem2

   type(point), intent(INOUT) :: v(natom)

   type(pointDP) dr
   type(point) dv
   real dp
!
   dr = r(mem2) - r(mem1)
   dv = v(mem2) - v(mem1)
   dp = -(dr%x * dv%x + dr%y * dv%y + dr%z * dv%z) / dotDP(dr, dr) / xsum
! modul del moment transferit en la colisio
   v(mem1)%x = v(mem1)%x - dp / xm(mem1) * dr%x
   v(mem1)%y = v(mem1)%y - dp / xm(mem1) * dr%y
   v(mem1)%z = v(mem1)%z - dp / xm(mem1) * dr%z

   v(mem2)%x = v(mem2)%x + dp / xm(mem2) * dr%x
   v(mem2)%y = v(mem2)%y + dp / xm(mem2) * dr%y
   v(mem2)%z = v(mem2)%z + dp / xm(mem2) * dr%z
 end subroutine chgmom
!===============================================================================
 subroutine updateTCol(i ,temps, r, v, intD, TMIN, natom)
! input r v de dues particules, output deltak i timp
 use geometry
 use geometryDP
 use stepPotentials
 use intList 
   integer, intent(IN) :: natom, i
   integer j
   type(pointDP), intent(IN) :: r(natom)
   type(point),intent(IN) :: v(natom)
   type(intData), intent(INOUT) :: intD
   real*8, intent(IN) :: temps
   real, intent(IN) :: TMIN

   integer k
   real*8 argk(MAXSTEPS), tijs(MAXSTEPS,2)
   real rij, rij2, vij, vij2, a,b, dotrv
   type(pointDP) dr
   type(point) dv
!
   integer lc(2)
!   
   j=intD%patnum
   intD%timp=1.
   intD%deltak=0.
!   
   dr = r(j) - r(i)
   dv = v(j) - v(i)
   rij2 = dotDP(dr,dr)
   rij = sqrt(rij2)
   vij2 = dot(dv,dv)
   vij = sqrt(vij2)
   dotrv = dr%x * dv%x + dr%y * dv%y + dr%z * dv%z
   tijs=0.
   if (rij2.le.intD%stepPt%step(intD%stepPt%nstep)%r**2 + dotrv**2/vij2) then
     do k = 1,intD%stepPt%nstep
        argk(k) = intD%stepPt%step(k)%r**2 - rij2 + dotrv**2/vij2
        if (argk(k).gt.0.) then
           a = -dotrv / vij2
           b = sqrt(argk(k)) / vij
           tijs(k,1) = a - b
           tijs(k,2) = a + b
        endif 
     enddo
     lc = minloc(tijs, mask=tijs.gt.TMIN)
     if (lc(1).gt.0.)  then
        intD%timp = temps + tijs(lc(1),lc(2))
        if (lc(2).eq.1) then
           intD%deltak =  intD%stepPt%step(lc(1))%e
        else
           intD%deltak = -intD%stepPt%step(lc(1))%e
        endif 
     endif
  endif
 end subroutine updateTCol
!===============================================================================
subroutine updateV (r, v, deltak, xm, xsum, mem1, mem2, natom)
 use geometry
 use geometryDP

   integer, intent(IN) :: natom, mem1, mem2
   type(pointDP), intent(IN) :: r(natom)
   real, intent(IN) :: deltak, xm(natom), xsum
   type(point), intent(INOUT) :: v(natom)

   type(pointDP) dr
   type(point) dv
   real rij, vdmod,  sto, dp
   real, parameter :: A2 = 1.e-20

   dr = r(mem2) - r(mem1)
   dv = v(mem2) - v(mem1)
! calculo vdmod, la projeccio de la diferencia de velocitats en l'eix que uneix les particules
   rij = sqrt(dotDP(dr,dr))
   vdmod = (dv%x * dr%x + dv%y * dr%y + dv%z * dr%z) / rij
! entra o surt d'un pou
   sto = vdmod**2 + 4. * deltak * xsum / A2
   if (sto.gt.0) then
! vario la velocitat
      dp = -vdmod + sign(1.,vdmod) * sqrt(sto)
      dp = dp / 2. / xsum / rij
   else
! les particules es queden atrapades al pou
      dp = -vdmod / xsum / rij 
!      write (4,*) "Rebot"
   endif
   v(mem1)%x = v(mem1)%x - dp / xm(mem1) * dr%x
   v(mem1)%y = v(mem1)%y - dp / xm(mem1) * dr%y
   v(mem1)%z = v(mem1)%z - dp / xm(mem1) * dr%z

   v(mem2)%x = v(mem2)%x + dp / xm(mem2) * dr%x
   v(mem2)%y = v(mem2)%y + dp / xm(mem2) * dr%y
   v(mem2)%z = v(mem2)%z + dp / xm(mem2) * dr%z
 end subroutine updateV 
!==============================================================================
subroutine nextCol(mem1, mem2, np1, np2, tevent, ipart, tpart, natom,nblist)
 use intList
   integer, intent(IN) :: natom
   integer, intent(IN) :: ipart(natom)
   real*8, intent(IN) :: tpart(natom)
   type (intpList), intent(IN) :: nblist(natom)
   integer, intent(OUT) :: mem1,mem2
   real*8, intent(OUT) :: tevent
   integer, intent(OUT) :: np1,np2
!
   mem1 = minloc(tpart,1)
   if (mem1.le.0) then
      write (0,*) "ERROR: No colision found (this should never happen!!)"
      stop 1
   endif
   tevent = tpart(mem1)
   np1 = ipart(mem1)
   mem2 = nblist(mem1)%iData(ipart(mem1))%patnum
   np2 = nblist(mem1)%iData(ipart(mem1))%simp
 end subroutine nextCol
!============================================================================================
 subroutine inici (nblist, tpart, ipart, natom)
 use intList
   integer, intent(IN) :: natom
   type (intpList), intent(IN) :: nblist(natom)
   real*8, intent(INOUT) :: tpart(natom)
   integer, intent(INOUT) :: ipart(natom)
   integer i,k
!
   tpart = 1.
   ipart = -1
   do i = 1,natom
   do k = 1,nblist(i)%nats
      if (nblist(i)%iData(k)%timp.lt.tpart(i)) then
         tpart(i) = nblist(i)%iData(k)%timp
         ipart(i) = k ! ipart recull index NO num d'atom
      endif
   enddo
   enddo   
 end subroutine inici
!=============================================================================================
 subroutine updateXocPart(m1, m2, nblist, temps, r, v, TMIN, natom, tpart, ipart)
 use intList
 use geometry
 use geometryDP
    integer, intent(IN) :: natom, m1, m2
    real, intent(IN) :: TMIN
    type(pointDP), intent(IN) :: r(natom)
    type(point), intent(IN) :: v(natom)
    type(intpList), intent(INOUT) :: nblist(natom)
    real*8, intent(IN) :: temps
    real*8, intent(INOUT) :: tpart(natom)
    integer, intent(INOUT) :: ipart(natom)
    integer j, k, l
!    
    tpart(m1) = 1.
    ipart(m1) = -1
!    
    do k = 1,nblist(m1)%nats
       j = nblist(m1)%iData(k)%patnum
       l = nblist(m1)%iData(k)%simp
       if (j.eq.m2) then
         nblist(m1)%iData(k)%timp=1.
         nblist(j)%iData(l)%timp=1.
       else       
          call updateTCol(m1, temps, r, v, nblist(m1)%iData(k), TMIN, natom)
!
          nblist(j)%iData(l)%timp = nblist(m1)%iData(k)%timp
          nblist(j)%iData(l)%deltak = nblist(m1)%iData(k)%deltak
! update tpart,ipart (m1)
          if (nblist(m1)%iData(k)%timp.lt.tpart(m1)) then
            tpart(m1) = nblist(m1)%iData(k)%timp
            ipart(m1) = k
          endif
! update tpart, ipart, parella
          if (nblist(j)%iData(k)%timp.lt.tpart(j)) then
            tpart(j) = nblist(m1)%iData(k)%timp
            ipart(j) = l
         endif
       endif
    enddo
 end subroutine updateXocPart
!===============================================================================   
 pure function calcSignCosAng (r, v, i, j, natom)
   use geometry
   use geometryDP
      real calcSignCosAng
      integer, intent(IN) :: natom, i, j
      type(pointDP), intent(IN) :: r(natom)
      type(point), intent(IN) :: v(natom)
      real :: dotv
      type(pointDP) dr
      type(point) dv

      dr = r(j) - r(i)
      dv = v(j) - v(i)
      dotv = dr%x * dv%x + dr%y * dv%y + dr%z * dv%z
      calcSignCosAng = sign(1.,dotv)
   end function calcSignCosAng     
!===============================================================================   
 pure function calcCM (natom, r, xm) result (rcm)
   use geometry
   use geometryDP
   type(point) rcm
   integer,intent(IN) :: natom
   type(pointDP), intent(IN) :: r(natom)
   real, intent(IN) :: xm(natom)
   integer i
   real xmassa
!
   rcm = point(0.,0.,0.)
   xmassa = sum(xm(1:natom))
   do i = 1,natom
      rcm%x = rcm%x + xm(i) * r(i)%x
      rcm%y = rcm%y + xm(i) * r(i)%y
      rcm%z = rcm%z + xm(i) * r(i)%z
   enddo
   rcm = (1./xmassa) * rcm
   end
!===============================================================================   

