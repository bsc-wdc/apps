!===============================================================================
   subroutine activateStepPot(stepPts, r, rcutcoul2, rcutsolv2, natom, nblist, xsum)
   use stepPotentials
   use geometryDP
   use intList
   integer, intent(IN) :: natom
   real, intent(IN) :: rcutcoul2, rcutsolv2
   type(pointDP), intent(IN) :: r(natom)
   type(stepPotInt), intent(INOUT) :: stepPts(natom,natom)
   type(intpList), intent(INOUT) :: nblist(natom)
   real, intent(IN) :: xsum(natom,natom)
   integer i, j
   real*8 rij2
   type(pointDP) rj
!
   where (stepPts%active)
      stepPts%active = .false.
   end where
   nblist%nats = 0
   do j = 2,natom
      rj = r(j) ! intentem millorar cache hits a r(i)
      do i = 1,j-1
!        if (stepPts(i,j)%tipInt.gt.SS) rij2 = calcDist2DP(r(i), r(j))
! inline 
!         if (stepPts(i,j)%tipInt.gt.SS) &
         rij2 = (r(i)%x-rj%x)**2+(r(i)%y-rj%y)**2+(r(i)%z-rj%z)**2
         if (stepPts(i,j)%tipInt.eq.SS.or.&
            (rij2.lt.rcutcoul2.and.stepPts(i,j)%tipInt.eq.COUL).or. &
            (rij2.lt.rcutsolv2.and.stepPts(i,j)%tipInt.gt.COUL)) then
            stepPts(i,j)%active = .true.
            nblist(i)%nats = nblist(i)%nats + 1
            nblist(j)%nats = nblist(j)%nats + 1
            nblist(i)%iData(nblist(i)%nats) = intData(j, nblist(j)%nats, stepPts(i,j), xsum(i,j), 1., 0.)
            nblist(j)%iData(nblist(j)%nats) = intData(i, nblist(i)%nats, stepPts(i,j), xsum(i,j), 1., 0.)
     endif            
  enddo
  enddo
 end subroutine activateStepPot
!===============================================================================
 subroutine thermalize(seed, iev, natom, TEMP, xmassa, v, xm, ekin)
 use geometry
   integer, intent(IN) :: iev, seed, natom
   real, intent(IN) ::  xmassa, temp, xm(natom)
   type(point), intent(OUT) :: v(natom)
   real, intent(OUT) :: ekin
   integer i, kk
   real sto, calcEkin
   type(point) vcm
   real*8 fi
!
   vcm = point(0., 0., 0.)
   do i = 1,natom
      kk = seed + 3 * i + 1 + iev
      call ran1(fi, kk)
      v(i)%x = fi - 0.5
      kk = seed + 3 * i + 2 + iev
      call ran1(fi, kk)
      v(i)%y = fi - 0.5
      kk = seed + 3 * i + 3 + iev
      call ran1(fi, kk)
      v(i)%z = fi - 0.5
!
      vcm = vcm + xm(i) * v(i)
   enddo
   vcm = (1./xmassa) * vcm
   do i = 1,natom
      v(i) = v(i) - vcm
   enddo
   sto = sqrt(1.5 * natom * TEMP / calcEkin(v, xm, natom))
   do i = 1,natom
      v(i) =  sto * v(i)
   enddo
   ekin = natom * TEMP ! No cal recalcularla 
 end subroutine thermalize
!========================================================================
   subroutine calcEpot (natom, r, stepPts, ego, epotgo, epotfis)
   use geometryDP
   use stepPotentials

   integer, intent(IN) :: natom
   type(pointDP), intent(IN) :: r(natom)
   type(stepPotInt), intent(IN) :: stepPts(natom,natom)
   real, intent(IN) :: EGO
   real, intent(OUT) :: epotgo, epotfis
   real dist
   integer i, j, k
!PENDENT TREBALLAR AMB NBLIST
   epotgo = 0.
   epotfis = 0.
   do j = 2,natom
   do i = 1,j-1
      dist = sqrt(calcDist2DP(r(i), r(j)))
      if (stepPts(i,j)%tipInt.eq.SS) then
         if (dist.gt.stepPts(i,j)%step(1)%r.and.dist.lt.stepPts(i,j)%step(2)%r) &
            epotgo = epotgo - ego
      elseif (stepPts(i,j)%active) then
         k = stepPts(i,j)%nstep
         do while ((k.gt.1).and.dist.lt.stepPts(i,j)%step(k)%r)
            epotfis = epotfis - stepPts(i,j)%step(k)%e / FACTE
            k = k - 1
         enddo
         if (dist.lt.stepPts(i,j)%step(k)%r) epotfis = epotfis - stepPts(i,j)%step(k)%e / FACTE
      endif
   enddo
   enddo
   end subroutine calcEpot
!========================================================================
 pure function  calcEkin (v, xm, natom) result (ekin)
 use geometry
   integer, intent(IN) :: natom
   real ekin
   type(point), intent(IN) :: v(natom)
   real, intent(IN) :: xm(natom)
   real, parameter :: a2 = 1.e-20
   integer i
   ekin = 0.
   do i = 1,natom
      ekin = ekin + 0.5 * xm(i) * a2 * dot(v(i), v(i))
   enddo
 end function calcEkin
!===============================================================================
 function distance(r,disttarg,w,ica,natom,nres,recnres) result (score)
 use geometryDP
   integer, intent(IN) :: natom, nres, recnres, ica(nres)
   type(pointDP), intent(IN) :: r(natom)
   real disttarg(nres,nres),w(nres,nres)
   real score, rij
   integer i,j
   score=0.
   do i=1,recnres-1
      do j=i+1,recnres
         rij = calcDistDP(r(ica(i)),r(ica(j)))
         score=score+w(i,j)*(rij-disttarg(i,j))**2
      enddo
   enddo
   do i=recnres+1,nres-1
     do j=i+1,nres
         rij = calcDistDP(r(ica(i)),r(ica(j)))
         score = score + w(i,j)*(rij-disttarg(i,j))**2
      enddo
   enddo
   return
 end function distance
!===============================================================================
 function MCCheck (seed, xbeta, scoreprev, score, score0) result (rej)
   integer, intent(IN) :: seed
   real, intent(IN) :: scoreprev,score,score0,xbeta
   logical rej
   real sto
   real*8 fi
   
   call ran1(fi, seed)
   sto=exp(xbeta*(scoreprev-score)/sqrt(score0*scoreprev))
   rej = (sto.lt.fi)
 end function MCCheck
!==============================================================================
