!===========================================================================
!  DISCRETE 0.2.4a
!
!  Discrete Molecular Dynamics, 
!
!   v. 0.1a    First public release. 
!      Read Topology and coordinates prepared by auxiliary program
!        Restart available
!   v. 0.2a    Fortran Free Form.
!      Dynamic Memory access
!   v. 0.2.3a  Optimization memory access, implemented ISOLV=0
!   v. 0.2.4a  Multilevel representation + DIMS
!
!========================================================================== 
   program discrete
   use commLine
   use geometry
   use geometryDP
   use stepPotentials
   use intList
   use paramSet
!   
#include "vars.h"
!
#ifdef DEBUG
   real ep0,ek0,et0,ep,ek,et
#endif
   call cpu_time(tinit)
   call readCommLine(files, unit_i, unit_o, NFILES)
   call readInputParamSet(unit_i)
!
   call programHeader(unit_o)
   call printFileSummary(files, unit_o)
   call writeInputParamSet(unit_o)
!
   write (unit_o, *)
   write (unit_o, '(" Reading system from topology file ")')
   write (unit_o, *)
! readTopology HO mantenim aqui per evitar problemes amb allocate
   unit_top = openFn(files, '-top')
   read(unit_top) tv  
   if (trim(tv).ne.trim(topVersion)) then
      write (0,*) "Error: incompatible topology, ", topVersion, " required"
      stop 1
   endif
   read(unit_top) molType
   write (unit_o,'(" Type of system:",i2)') molType
   read(unit_top) nmol, nres, natom, recnatom
   write (unit_o, '(" Molecules: "i4," Residues:   ",i4," Atoms:  ",i6)') nmol,nres, natom
   allocate (mols(nmol, 2), stat=ioerr)
   call errorAllocmem (ioerr, 'NMol')
   allocate (in(nres), ica(nres), ico(nres), molres(nres), stat=ioerr)
   call errorAllocmem (ioerr, 'NRes')
   allocate (atom(natom), rnum(natom), res(natom), atp(natom), qq(natom),  gfree(natom), &
             vol(natom), evdw(natom), rvdw(natom), rhc(natom), xm(natom), xlamb(natom), &
             molnum(natom), dummy(natom), frozen(natom), nxm(natom), chain(natom), stat=ioerr)
   call errorAllocmem (ioerr, 'NAtom')
!
   read(unit_top) (mols(i,1),mols(i,2),i=1,nmol)
   read(unit_top) (molres(i),i=1,nres)
   if (molType.eq.PROT.or.molType.eq.COMPLEX) read (unit_top) (in(i), ica(i), ico(i), i=1,nres)
!
   read (unit_top) (atom(i), rnum(i), res(i), chain(i), atp(i), &
                     molnum(i), frozen(i), dummy(i), i=1,natom)
!
   recnmol=molnum(recnatom)
   recnres=rnum(recnatom)
   write(unit_o,*)moltype
   if(molType.eq.COMPLEX) &
 write (unit_o, '(" RECEPTOR: Molecules: "i4," Residues:   ",i4," Atoms:  ",i6)') recnmol, recnres, recnatom
 
   do i=1,natom
      read (unit_top) qq(i), gfree(i), vol(i), evdw(i), rvdw(i), rhc(i), xm(i)
   enddo
!   xmassa = sum(xm) ! mogut a llista real
   xlamb = 3.5
!
   read (unit_top) ncovpairs, nhbs, j 
   write (unit_o, '(" Cov. Pairs:",i7," H. Bonds: ",i4)') ncovpairs, nhbs
!
   allocate (cov(ncovpairs,2), hbs(nhbs,2), rcov2(ncovpairs), stat=ioerr)
   call errorAllocmem (ioerr, 'Cov & Hbs')
!
   read (unit_top) (cov(i,1), cov(i,2), rcov2(i), i=1,ncovpairs)
   read (unit_top) (hbs(i,1), hbs(i,2), i=1,nhbs)
   if (molType.eq.PROT.or.molType.eq.COMPLEX) then
      read (unit_top) nhelix
      write (unit_o, '(" Num. Helix:       ",i4)') nhelix
!
      allocate (helix(nhelix,2), stat=ioerr)
      call errorAllocmem (ioerr, 'Helix')
      read (unit_top) (helix(i,1), helix(i,2), i=1,nhelix)
!
      read (unit_top) nbeta
      write (unit_o, '(" Num. Beta Strands:",i4)') nbeta
!
      allocate (beta(nbeta,2), stat=ioerr)
      call errorAllocmem (ioerr, 'Beta')
!
      read (unit_top) (beta(i,1), beta(i,2), i=1,nbeta)
   endif
   close (unit_top)
! end readTopology
!
! readCoordinates
   unit_r = openFn(files, '-r')
   read (unit_r) j
   if (j.ne.natom) then
      write (0,*) "ERROR: coordinates and topology files do not match"
      stop 1
   endif
! modificat per eliminar dummies
   allocate (rsp(natom), rorig(natom), nid(natom), r(natom), oid(natom), stat=ioerr)
   call errorAllocmem (ioerr, 'Coordinates')
   read (unit_r) (rsp(i)%x, rsp(i)%y, rsp(i)%z, i=1,natom)
   close (unit_r)
! end readCoordinates
   write (unit_o, '(" Initial coordinates read in ")')
   write (unit_o,*)
!
!      open(unit=10,file='input.pdb')
      unit_input = openFn(files, '-in')
      do i = 1,natom
         write (unit_input, '(A4,2X,I5,2X,A3,1X,A3,1X,A1,1X,I3,4X,3F8.3)') 'ATOM',i,atom(i),res(i),chain(i),rnum(i), rsp(i)
      enddo
      close(unit_input)
!
   if (idims.eq.1)then
      allocate (rtarg(natom), rprev(natom), rtargorig(natom),icatarg(natom),atomtg(natom),restg(natom))
      write (unit_o,'(" DIMS: Steered dynamics activated")')
      write (unit_o,'("   Re-scoring time (ps): ", f8.2)') trect/1e3
      write (unit_o,'("   MC Factor:            ", f8.2)') xbeta
      write (unit_o,'("   Convergence Lim:      ", f8.2)') sclim
      write (unit_o,*)
      unit_targ = openFn(files, '-targ')
      i = 1
200     read (unit_targ, '(A4,2X,I5,2X,A3,1X,A3,3X,I3,4X,3F8.3)', end=201) &
           c1,j,atomtg(i),restg(i),k,rtargorig(i)
        i=i+1
      goto 200
201      close(unit_targ)
      write (unit_o, '("  Target structure read in")')
!     allocate (disttarg(nres,nres),w(nres,nres))
!     do i=1,nres-1
!     do j=i+1,nres
!        disttarg(i,j) = calcDistDP(rtarg(ica(i)),rtarg(ica(j)))
!        pdistab       = 0.5*(calcDistDP(r(ica(i)),r(ica(j))) + disttarg(i,j))
!        w(i,j)        = ((calcDistDP(r(ica(i)),r(ica(j))) - disttarg(i,j))**2)/pdistab**4
!     enddo
!     enddo
   endif   
!
! Tractament de dummies, creem la llista efectiva d'atoms i mantenim punters a la numeracio original
!
   nat=0
   nid=0
   oid=0
   do i = 1,natom
      rorig(i) = SPtoDP(rsp(i))
      if (.not.dummy(i)) then
         nat=nat+1
         r(nat) = rorig(i)
         nid(i) = nat
         oid(nat) = i
         nxm(nat) = xm(i)
         if (idims.eq.1) rtarg(nat) = rtargorig(i)
      endif
   enddo
    if(idims.eq.1)then
! update position of CA with the multiscale mapping
      k=1
      do i=1,natom
      j=nid(i)
   if(atom(i).eq.'CA')then
   ica(k)=j
   k=k+1
   endif
      enddo
      allocate (disttarg(nres,nres),w(nres,nres))
! parameters for the scoring function
! receptor
      do i=1,recnres-1
      do j=i+1,recnres
         disttarg(i,j) = calcDistDP(rtarg(ica(i)),rtarg(ica(j)))
         pdistab       = 0.5*(calcDistDP(r(ica(i)),r(ica(j))) + disttarg(i,j))
         w(i,j)        = ((calcDistDP(r(ica(i)),r(ica(j))) - disttarg(i,j))**2)/pdistab**4
      enddo
      enddo
! ligand
      do i=recnres+1,nres-1
      do j=i+1,nres
         disttarg(i,j) = calcDistDP(rtarg(ica(i)),rtarg(ica(j)))
         pdistab       = 0.5*(calcDistDP(r(ica(i)),r(ica(j))) + disttarg(i,j))
         w(i,j)        = ((calcDistDP(r(ica(i)),r(ica(j))) - disttarg(i,j))**2)/pdistab**4
      enddo
      enddo
   endif
   xmassa = sum(nxm(1:nat))
   if (natom.ne.nat) write (unit_o,'(" Total atoms corrected considering dummies", i7)') nat
   onatom=natom
   natom=nat
! A partir d'aqui nomes treballem 
   allocate (distat2(natom,natom))
   do j = 2, natom
   do i = 1, j-1
      distat2(i,j) = calcDist2DP(r(i), r(j))
      distat2(j,i) = distat2(i,j)
   enddo
   enddo        
!
   allocate (v(natom), rb(natom,natom), drb(natom,natom), istruct(natom,natom), stepPts(natom,natom), &
      xsum(natom,natom), stat=ioerr)
   call errorAllocmem(ioerr, 'Setup/Natom')
   if (RST.eq.1) then ! input velocities
      unit_v = openFn(files, '-v')
      read (unit_v) j
      if (j.ne.natom) then
         write (0,*) "ERROR: velocities and topology files do not match"
         stop 1
      endif
      read (unit_v) (v(i)%x, v(i)%y, v(i)%z, i=1,natom)
      close (unit_v)
   endif
!      
   rb = 0.
   drb = 0.
   istruct = .false.
   stepPts%active = .false.
!
   do i = 1,ncovpairs
      rb(nid(cov(i,1)),nid(cov(i,2))) = sqrt(rcov2(i))
      drb(nid(cov(i,1)),nid(cov(i,2))) = SIGMA*rb(nid(cov(i,1)),nid(cov(i,2)))
   enddo
   do i = 1,nhbs
      if (atom(hbs(i,1)).eq.'N'.or.atom(hbs(i,2)).eq.'N') then
         rb (nid(hbs(i,1)),nid(hbs(i,2))) = 0.5 * (rnomax + rnomin)
         drb(nid(hbs(i,1)),nid(hbs(i,2))) = 0.5 * (rnomax - rnomin)
      else
         rb (nid(hbs(i,1)),nid(hbs(i,2))) = 0.5 * (rcomax + rcomin)
         drb(nid(hbs(i,1)),nid(hbs(i,2))) = 0.5 * (rcomax - rcomin)
      endif
   enddo
! fixa els angles diedrics dels residus que formen part d'un element d'estructura secundaria
   if(IDAB.eq.1)then
      do i = 1,nhelix
         do j = helix(i,1)+1,helix(i,2)
            if (nid(ico(j-1)).gt.0.and.nid(ico(j)).gt.0) &
               istruct(nid(ico(j-1)),nid(ico(j))) = .true.
         enddo
         do j = helix(i,1),helix(i,2)-1
            if (nid(in(j)).gt.0.and.nid(in(j+1)).gt.0) &
               istruct(nid(in(j)),nid(in(j+1))) = .true.
            do k = j+1,helix(i,2)
               istruct(nid(ica(j)),nid(ica(k))) = .true.
            enddo
         enddo
      enddo
      do i = 1,nbeta
         do j = beta(i,1)+1,beta(i,2)
            if (nid(ico(j-1)).gt.0.and.nid(ico(j)).gt.0) &
               istruct(nid(ico(j-1)),nid(ico(j))) = .true.
         enddo
         do j = beta(i,1),beta(i,2)-1
            if (nid(in(j)).gt.0.and.nid(in(j+1)).gt.0) &
               istruct(nid(in(j)),nid(in(j+1))) = .true.
            do k = j+1,beta(i,2)
                if (distat2(nid(ica(j)),nid(ica(k))).lt.RCUTGO**2) then
                  istruct(nid(ica(j)),nid(ica(k))) = .true.
               endif
            enddo
         enddo   
      enddo
   endif   
!
   do j = 2, natom
   do i = 1, j-1
      if (istruct(i,j)) stepPts(i,j) = getStepSSec(SIGMAGO, EGO, sqrt(distat2(i,j)), .true.)
   enddo
   enddo
!
   do j = 2,natom
   do i = 1, j-1 
      if (.not.istruct(i,j).and.rb(i,j).lt.1.e-20)then
         esolv = -sqrt(evdw(oid(i)) * evdw(oid(j))) * FVDW
         rvdwij = rvdw(oid(i)) + rvdw(oid(j)) - SVDW
         potlk = gfree(oid(i)) * vol(oid(j)) / xlamb(oid(i)) + gfree(oid(j)) * vol(oid(i)) / xlamb(oid(j))
         esolv = esolv - potlk / rvdwij**2 * 0.09 * FSOLV
         ecoul = qq(oid(i))*qq(oid(j)) * EPS ! Si es constant dielectrica millor qq2/EPS!!
         if(abs(qq(oid(i))*qq(oid(j))).gt.1d-10.and.EPS.gt.1.d-10)then
            stepPts(i,j) = getStepCoul (rvdwij, DPSINT, DPSEXT, ecoul, HPS, .false.)     
         else
            if(esolv.lt.0)then
!               if(rnum(oid(j)).gt.(rnum(oid(i))+1)) stepPts(i,j) = getStepHFob(rvdwij, DHF, esolv, .false.)
               stepPts(i,j) = getStepHFob(rvdwij, DHF, esolv, .false.)
            else
               stepPts(i,j) = getStepHFil(rvdwij, DHF, esolv, .false.)
            endif
         endif
      endif
!
      xsum(i,j) = 0.5 * (1./nxm(i) + 1./nxm(j))      
   enddo
   enddo
! precalcul parells
   bpairs = count(rb.gt.1.e-20)
   allocate(blist(bpairs,2), stat = ioerr)
   call errorAllocMem(ioerr, 'Bond pairs')
   bpairs=0
   do j = 2,natom
   do i = 1, j-1 
      if (rb(i,j).gt.1e-20) then
         bpairs = bpairs + 1
         blist(bpairs,1) = i
         blist(bpairs,2) = j
         stepPts(i,j)%tipInt = BOND
         stepPts(i,j)%active = .false.
      endif
   enddo
   enddo
!
   allocate(nblist(natom), stat=ioerr)
   call errorAllocMem(ioerr, 'Non Bonded List 1')
   do i = 1,natom
        nblist(i) = allocateintPList(natom, ioerr)
        call errorAllocMem(ioerr, 'Non Bonded List')
   enddo
   call activateStepPot(stepPts, r, rcutcoul2, rcutsolv2, natom, nblist, xsum)
!
   write (unit_o, '(" Initial Pairs list")')
   write (unit_o, '(" ==================")')
   write (unit_o, '(" Total:              ",i9)') natom * (natom - 1) / 2
   write (unit_o, '(" Bonded:             ",i9)') bpairs
   write (unit_o, '(" Non Bonded:         ",i9)') count(stepPts%active)
   write (unit_o, '("   Secondary Struc.  ",i9)') count(stepPts%tipInt.eq.SS)
   write (unit_o, '("   Electrostatic     ",i9)') count(stepPts%tipInt.eq.COUL.and.stepPts%active)
   write (unit_o, '("   Hydrophilic       ",i9)') count(stepPts%tipInt.eq.HFil.and.stepPts%active)
   write (unit_o, '("   Hydrophobic       ",i9)') count(stepPts%tipInt.eq.HFob.and.stepPts%active)
! 
   write (unit_o, *) 
   write (unit_o,'(" System setup completed")')
   write (unit_o,*)
! Deallocate
   deallocate (in, ico, atp,qq, gfree, cov, hbs, helix, beta, distat2)
   deallocate (istruct)
!
   call cpu_time(tsetup)
! Posem temps en segons
   TSNAP = TSNAP * 1.e-15
   TCORR = TCORR * 1.e-15
   TRECT = TRECT * 1.e-15
   TACT = TACT * 1.e-15
! suma l'energia potencial de la conformacio inicial
   call calcEpot (natom, r, stepPts, EGO, epotgo, epotfis)
   rcm = calcCM (natom, r, nxm)
!
   if (OUTFORMAT.eq.1) then
      unit_pdb = openFn(files, '-traj')
      unit_traj = openFn(files, '-x')
   else
      unit_traj = openFn(files, '-traj')
   endif
!   unit_rst =  openFn(files, '-rst')
!   if (RST.eq.1) unit_vrst =  openFn(files, '-rstv')      
   unit_ener = openFn(files, '-ener')
!
   do i = 1,natom
      r(i) = r(i) - SPtoDP(rcm)
      if (idims.eq.1) rprev(i) = r(i)
   enddo
   if (OUTFORMAT.eq.1) then
      call writeSnapshot (unit_pdb, ibloc, r, atom, res, chain, rnum, natom, oid,onatom, 0)
      write (unit_traj,'(a132)') "(10f8.3)"
   else
      call writeSnapshot (unit_traj, ibloc, r, atom, res, chain, rnum, natom, oid,onatom, OUTFORMAT)
   endif
!
   iev=0
   if (RST.eq.0) call thermalize(seed, iev, natom, temp, xmassa, v, nxm, ekin)
   write (unit_o, '(" Initial energy evaluation completed")')
   write (unit_o, '(" Epot (kcal/mol) = ",f10.3," Ekin (kcal/mol) = ",f10.3)') epotfis, ekin / FACTE
!   if (iwr.eq.1) &
   write (unit_ener, '("# Initial energy ",4f10.3)') epotfis, epotgo, ekin/FACTE, &
   epotfis+epotgo+ekin/FACTE
!   if (iwr.eq.1) & 
   write (unit_ener, '("# time, epotFis, epotGo, ekin0, ekin1, Etot, Temp")')
   temps = 0.
   tactot = 0.
      iacc=0
      idec=0
      iaccp=0
      idecp=0
         if (idims.eq.0) then
            write (unit_ener,'(1x,7f8.2)') temps*1.e12, epotfis, epotgo, ekin0/FACTE, ekin/FACTE, &
                  epotfis+epotgo+ekin0/FACTE,ekin0/(1.5 * natom)
         else 
            write (unit_ener,'(1x,11f8.2)') tactot*1.e12, temps*1.e12, epotfis, epotgo, ekin0/FACTE, ekin/FACTE, &
                  epotfis+epotgo+ekin0/FACTE,ekin0/(1.5 * natom), iaccp*1./idecp, iacc*1./idec, score
         endif
   
   write (unit_o,*)
   allocate (tpart(natom), ipart(natom), stat=ioerr)
   call errorAllocMem(ioerr, ' Colision times')
   score0 = 1.
   if (IDIMS.eq.1) then
      write (unit_o,*)'DIMS'
      score0 = distance(r, disttarg, w, ica, natom, nres, recnres)
      scoreprev=score0
      iacc=0
      idec=0
      write (unit_o, '(" Initial scoring: ",f10.2)') score0
      if (SCLIM.gt.0) write (unit_o, '(" Target scoring:  ", f10.2)') score0*sclim
      write (unit_o,*)
   endif
   tpart = 1.
   ipart = -1
   score=score0
   ibloc=1
!----------------------------------------------------------------------------
   do while (ibloc.le.NBLOC.and.(score/score0).gt.SCLIM)
      tacum = 0.
      iaccp = 0.
      idecp = 0.
!----------------------------------------------------------------------------
      do while(tacum.lt.TSNAP)
         tacrect=0
         do while (tacrect.lt.TRECT)
            taccorr = 0.
            iev = 0
            ierr = 0
!----------------------------------------------------------------------------
            do while(taccorr.lt.TCORR)
               call colisioBond(bpairs, blist, r, v, rb, drb, nxm, xsum, natom)
               call activateStepPot(stepPts, r, rcutcoul2, rcutsolv2, natom, nblist, xsum)
               call colisioNonBond(stepPts, temps, nblist, r, v, rhc, shc, rnum, atom, nxm, ierr, &
                                TMIN, natom, ISOLV)
               call inici (nblist, tpart, ipart, natom) !tpart(i) ipart(i) contenen la primera col per atom
               tacact = 0.
! evolucio temporal
!------------------------------------------------------------------------------
            do while (tacact.lt.TACT)
! busca quina es la peopera colisio
               tevent = 1.
               call nextCol(mem1, mem2, npair1, npair2, tevent, ipart, tpart, natom, nblist) 
               tevent1 = tevent - temps
               temps = tevent
               tacact = tacact + tevent1
               iev = iev + 1
#ifdef DEBUG
               call calcEpot (natom, r, stepPts, ego, epotgo, epotfis)
               ep0=epotgo+epotfis
               ek0 = calcEkin(v,nxm,natom) 
               et0 = ep0 + ek0/FACTE
#endif
! translacio i variacio dels temps
               do i=1,natom ! mantenim la versio inline degut a la barreja de tipus de real !!
                  r(i)%x = r(i)%x + v(i)%x * tevent1
                  r(i)%y = r(i)%y + v(i)%y * tevent1
                  r(i)%z = r(i)%z + v(i)%z * tevent1
               enddo
               call updateV(r, v, nblist(mem1)%iData(npair1)%deltak, nxm, nblist(mem1)%iData(npair1)%xsum, mem1, mem2, natom)
!
                  r(mem1)%x = r(mem1)%x + v(mem1)%x * tevent1/100.
                  r(mem1)%y = r(mem1)%y + v(mem1)%y * tevent1/100.
                  r(mem1)%z = r(mem1)%z + v(mem1)%z * tevent1/100.
                  r(mem2)%x = r(mem2)%x + v(mem2)%x * tevent1/100.
                  r(mem2)%y = r(mem2)%y + v(mem2)%y * tevent1/100.
                  r(mem2)%z = r(mem2)%z + v(mem2)%z * tevent1/100.
!
#ifdef DEBUG
              call calcEpot (natom, r, stepPts, ego, epotgo, epotfis)
              ek = calcEkin(v,nxm,natom)   
              ep=epotgo+epotfis
              et = ep + ek/FACTE
              write (unit_ener,'(a1,5f10.3,3f15.4)') "D",temps*1.e15, epotfis, epotgo, ek/FACTE,  &
                    ep-ep0+ (ek-ek0)/FACTE, ep-ep0, (ek-ek0)/FACTE, nblist(mem1)%iData(npair1)%deltak/FACTE
#endif
! ara calcula els temps de colisio per a les dues particules que han xocat
               call updateXocPart(mem1, mem2, nblist, temps, r, v, TMIN, natom, tpart, ipart)
               call updateXocPart(mem2, mem1, nblist, temps, r, v, TMIN, natom, tpart, ipart)
               enddo
! end do while (tacact.lt.tact)------------------------------------------------
               taccorr = taccorr + tacact
            enddo
! end do while(taccorr.lt.tcorr)-----------------------------------------------
            ekin0 = calcEkin(v,nxm,natom)
            if (TCALC.eq.1) call thermalize (seed, iev, natom, temp, xmassa, v, nxm, ekin)
            tacrect = tacrect + taccorr
         enddo
! end DO while(tacrect.lt.trect) -------------------------------------------------
         if (IDIMS.eq.1) then
            score = distance(r,disttarg,w,ica,natom,nres, recnres)
            idec=idec+1
            idecp=idecp+1
            if (MCCheck(seed+iev, xbeta, scoreprev, score, score0)) then
               r=rprev
            else
               rprev=r
               iacc=iacc+1
               iaccp=iaccp+1
               scoreprev=score
               tacum=tacum+tacrect
               tactot=tactot+tacrect
            endif
         else
            iacc=iacc+1
            iaccp=iaccp+1
            tacum=tacum+tacrect
            tactot=tactot+tacrect
         endif
      enddo
! end DO while(tacum.lt.tsnap) -------------------------------------------------
      call calcEpot (natom, r, stepPts, ego, epotgo, epotfis)
!      if (iwr.eq.1) then
         if (idims.eq.0) then
            write (unit_ener,'(1x,7f8.2)') temps*1.e12, epotfis, epotgo, ekin0/FACTE, ekin/FACTE, &
                  epotfis+epotgo+ekin0/FACTE,ekin0/1.5/natom
         else 
            write (unit_ener,'(1x,11f8.2)') tactot*1.e12, temps*1.e12, epotfis, epotgo, ekin0/FACTE, ekin/FACTE, &
                  epotfis+epotgo+ekin0/FACTE,ekin0/1.5/natom, iaccp*1./idecp, iacc*1./idec, score
         endif
!      endif
      rcm = calcCM(natom, r, nxm) 
      call writeSnapShot (unit_traj, ibloc, r, atom, res, chain, rnum, natom, oid, onatom, OUTFORMAT)
!      if (iwr.eq.1) then
         if (idims.eq.0) then
            write (unit_o, &
         '(" Time (ps): ",f8.2," Events: ", i4, " Overlaps ",i4, " EPot (kcal/mol): ",2f10.3," Temp (K):" f10.2)') &
                           temps*1.e12, iev, ierr, epotfis, epotgo, ekin0/1.5/natom
         else
            write (unit_o, &
         '(" Time (ps): ",2f8.2," Events: ", i4, " Overlaps ",i4, " EPot (kcal/mol): ",2f10.3," Temp (K):" 4f10.2)') & 
                           tactot*1.e12, temps*1.e12, iev, ierr, epotfis, epotgo, ekin0/1.5/natom, &
                            iaccp*1./idecp, iacc*1./idec, score
         endif                               
!      endif                         
#ifdef DEBUG
      write (unit_o, '(" NB pairs (SS, COUL, HFil, HFob):",5i7)') count(stepPts%active), & 
         count(stepPts%tipInt.eq.SS), & 
         count(stepPts%tipInt.eq.COUL.and.stepPts%active), & 

         count(stepPts%tipInt.eq.HFil.and.stepPts%active), &
         count(stepPts%tipInt.eq.HFob.and.stepPts%active)
#endif
! restart
      do i = 1,natom
         rsp(i) = DPtoSP (r(i))
      enddo
      unit_rst =  openFn(files, '-rst')
      write (unit_rst) natom
      write (unit_rst) (rsp(i), i = 1,natom)
      close(unit_rst)
      if (RSTV.eq.1) then 
         unit_rstv =  openFn(files, '-rstv')      
         write (unit_rstv) natom
         write (unit_rstv) (v(i), i = 1,natom)      
         close(unit_rstv)
      endif         
!      
      ibloc=ibloc+1         
   enddo
! end DO iblock----------------------------------------------------------------- 
   close(5)
   rcm = calcCM (natom,r,nxm)
!   do i = 1,natom
!      rsp(i) = DPtoSP (r(i))
!   enddo
!   write (unit_rst) natom
!   write (unit_rst) (rsp(i), i = 1,natom)
   call cpu_time(tfin)
   write (unit_o, '(" T I M I N G S ")')
   write (unit_o, '(" ============= ")') 
   write (unit_o, '(" Setup: " f10.2," s")') tsetup-tinit
   write (unit_o, '(" Traj:  ",f10.2," s (",f10.2," ns/h, ",f7.2," ns/day)")') tfin-tsetup, &
      tactot*1.e15 / (tfin-tsetup) * 3.6 / 1000. , tactot*1.e15 / (tfin-tsetup) * 3.6 / 1000. * 24
   write (unit_o, *)
   end
!=============================================================================

