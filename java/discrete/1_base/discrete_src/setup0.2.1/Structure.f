!
! Authors : Josep Ll. Gelpi, Agusti Emperador
! Subject : Structure module
! Revision: $id$
!
 MODULE Structure
 USE geometry
!
 type pairlist
   integer npairs
   integer, pointer :: list (:,:)
 end type pairlist
 
 type Fragment
   integer ini, fin
 end type Fragment
 
 type atomData
   character(len=4) atomId
   character(len=4) resIdbyAtom
   integer resNum ! ref to residue num per atom, unique values
   integer resNumPDB ! ref to PDB residue num per atom
   integer molNum ! ref to mol num per atom
   character(len=4) atType
   character(len=1) chainId
   type(point) r
   real charge
   logical frozen ! tied by pseudo-cov bonds to keep frozen
   logical dummy ! atoms to be ignored during simulation  
 end type atomData
 
 type residueData
   character(len=4) resId
   integer resNumPDB
   character(len=1) chainId
   integer ini,fin,in,ica,ico,molres ! atom pointers per residue
 end type residueData

 type struc 
   integer natoms, nres, nmol
   type(atomData), pointer :: ats(:)
   type(residueData), pointer :: res(:)
   integer*1, pointer :: bonds(:,:)
   real, pointer :: distat2(:,:)
   type (pairList) cov, hbs, other
   integer nhelix,nbeta
   type (Fragment), pointer :: helix(:), beta(:) ! ini and fin RESIDUE per fragment
   type (Fragment), pointer :: mols(:) ! ini and fin RESIDUE per fragment
   integer moltype
 end type struc
!
 real, parameter :: RSSMAX = 2.5
 real, parameter :: RNOMAX = 4.1
 real, parameter :: RNOMIN = 2.5
 real, parameter :: RNCMAX = 5.
 real, parameter :: RNCMIN = 3.2
 real, parameter :: RCOMAX = 5.
 real, parameter :: RCOMIN = 3.2
!
 integer, parameter :: COV=1, HB=2
 integer, parameter :: PROT=1, NUC=2, SMALL=3, COMPLEX=4 ! molType
 integer, parameter :: ALL=0, HEAVY=1, CAONLY=2 ! CGTYpe
 integer, parameter :: MD=0, DOCKING=1 ! tipCalc
!
 character(len=10), parameter :: topVersion = 'v0.2.1.2'
!
 character(len=50), parameter :: pdbinputFmt = '(13X,A3,1X,A4,A1,I4,4X,3F8.3,2f6.2)'
!===============================================================================
 CONTAINS
!===============================================================================
 function findAtominRes(str,ires,atomId) result (p)
 use utils
   integer p
   type(struc), intent(IN) :: str
   integer, intent(IN) :: ires
   character(*), intent(IN) :: atomId
!
   p=str%res(ires)%ini
   do while (p.lt.str%res(ires)%fin.and..not.eqId(str%ats(p)%atomId,atomId))
      p=p+1
   enddo
   if (.not.eqId(str%ats(p)%atomId,atomId)) p=0

 end function findAtominRes
!===============================================================================
 function allocatePairList(npairs) result (pl)
   type(pairList) pl
   integer, intent(IN) :: npairs
   allocate (pl%list(npairs,2))
   pl%npairs=npairs
   pl%list = 0
 end function allocatePairList
!===============================================================================
 function allocateStructureAtoms (natoms) result (st)
   type(struc) st
   integer, intent(IN) :: natoms
   allocate (st%ats(natoms))
   allocate (st%bonds(natoms,natoms), st%distat2(natoms,natoms))
   st%ats%atomId=''
   st%ats%resIdbyAtom=''
   st%ats%chainId=''
   st%ats%resnum=0
   st%ats%resnumPDB=0
   st%ats%molnum=0
   st%ats%r=point(0,0,0)
   st%ats%atType=''
   st%ats%charge=0.
   st%ats%frozen=.false.
   st%ats%dummy=.false.
   st%bonds=0
   st%distat2=0
   st%moltype=PROT
   st%natoms=natoms
 end function allocateStructureAtoms
!=============================================================================== 
 subroutine allocateStructureResidues(st, nres)
   type(struc) st
   integer, intent(IN) :: nres
   allocate(st%res(nres))
   st%res%resId=''
   st%res%resNumPDB=0
   st%res%chainId=''
   st%res%molres=0
   st%res%ini=99999
   st%res%fin=0
   st%res%in=0
   st%res%ica=0
   st%res%ico=0
   st%nres=nres
 end subroutine allocateStructureResidues
!===============================================================================
 subroutine allocateStructureMols(st, nmol)
   type(struc) st
   integer, intent(IN) :: nmol
   allocate(st%mols(nmol))
   st%nmol=nmol
   st%mols%ini=99999
   st%mols%fin=0
 end subroutine allocateStructureMols
!===============================================================================
 function loadStructurePDB (unt) result (st)
 USE utils
   type(struc) st
   integer, intent(IN) :: unt
   integer i,j,refres, nmol, nres, natoms
   character(len=1) refch
   character(len=80) str
!
   natoms=0
   nmol=0
!
  refch=''  
10 read(unt,'(a80)', end=20) str
! ELIMINEM EL H per posicio, pendent repasar canvis de format
      if (validAtomLine(str)) natoms=natoms+1
      if (terStr(str).or.(str(22:22).ne.refch).or.nmol.eq.0) then
         nmol=nmol+1
         refch = str(22:22)
      endif
      goto 10
20 continue
   st = allocateStructureAtoms(natoms)
   call allocateStructureMols(st,nmol)
!
   rewind(unt)
!
   i=0
   natoms=0
   nmol=0
   nres=0
   refch=''
   refres=0
30   read(unt,'(a80)', end=40) str
     if (terStr(str).and.nmol.gt.0) then
        if (st%mols(nmol)%fin.ne.0) nmol=nmol+1        
     endif
     if (validAtomLine(str)) then 
!       write (6,*) str
      i=i+1
      read (str, pdbinputFmt, end=40) st%ats(i)%atomId,st%ats(i)%resIdbyAtom,st%ats(i)%chainId,st%ats(i)%resNumPDB,&
      st%ats(i)%r%x,st%ats(i)%r%y,st%ats(i)%r%z,st%ats(i)%charge
      if (refch.ne.st%ats(i)%chainId.or.nmol.eq.0) then ! salvem PDBs sense cadena
        if (nmol.eq.0) then
          nmol=nmol+1
        elseif (st%mols(nmol)%fin.ne.0) then
         nmol=nmol+1        
        endif
      endif
      st%ats(i)%molNum = nmol
      if (refch.ne.st%ats(i)%chainId.or.refres.ne.st%ats(i)%resnumPDB) nres=nres+1
      refch=st%ats(i)%chainId
      refres=st%ats(i)%resnumPDB
      st%ats(i)%resNum=nres
      st%mols(nmol)%ini = min (st%ats(i)%resNum,st%mols(nmol)%ini)
      st%mols(nmol)%fin = max (st%ats(i)%resNum,st%mols(nmol)%fin)
      natoms=natoms+1
     endif
   goto 30    
40 continue
! cas de TER doblat al final
   do while (st%mols(nmol)%fin.eq.0)
       nmol=nmol-1
   enddo       
   st%nmol=nmol
   do i=1,st%natoms-1
   do j=i+1,st%natoms
      st%distat2(i,j) = calcDist2(st%ats(i)%r,st%ats(j)%r)
      st%distat2(j,i) = st%distat2(i,j)
   enddo
   enddo
!
   call allocateStructureResidues(st, nres)
!
   do i=1,st%natoms
     st%res(st%ats(i)%resNum)%resId=st%ats(i)%resIdbyAtom
     st%res(st%ats(i)%resNum)%resNumPDB=st%ats(i)%resNumPDB
     st%res(st%ats(i)%resNum)%chainId=st%ats(i)%chainId
     st%res(st%ats(i)%resNum)%ini = min(i,st%res(st%ats(i)%resNum)%ini)
     st%res(st%ats(i)%resNum)%fin = max(i,st%res(st%ats(i)%resNum)%fin)
     st%res(st%ats(i)%resNum)%molres = st%ats(i)%molNum
     if (eqId(st%ats(i)%atomId, 'N'))  st%res(st%ats(i)%resNum)%in = i
     if (eqId(st%ats(i)%atomId, 'CA')) st%res(st%ats(i)%resNum)%ica = i
     if (eqId(st%ats(i)%atomId, 'C'))  st%res(st%ats(i)%resNum)%ico = i
   enddo
! PENDENT determinar tipus de molecula 
  st%molType=PROT
 end function loadStructurePDB
!===============================================================================
 logical function validAtomLine(str)
   character(*), intent(IN) :: str
   validAtomLine = (str(1:4).eq.'ATOM'.or.str(1:4).eq.'HETA').and. &
   (str(14:14).ne.'H').and.(str(13:13).ne.'H')
 end function validAtomLine
!===============================================================================
 logical function terStr(str)
   character(*), intent(IN) :: str
   terStr = (str(1:3).eq.'TER')
 end function terStr
!===============================================================================
! prepares newStr as two succesive loadStructurePDB
 function mergeStructures (str1, str2, x,y,z) result (newStr)
 use geometry
  type(struc), intent(IN) :: str1, str2
  type(struc) newStr
  real, intent(IN):: x,y,z
  integer i,j
!
  newStr = allocateStructureAtoms(str1%natoms+str2%natoms)
!
  newStr%ats(1:str1%natoms) = str1%ats(1:str1%natoms)
!  
  newStr%distat2(1:str1%natoms,1:str1%natoms) = str1%distat2(1:str1%natoms,1:str1%natoms)
!
  newStr%ats(1+str1%natoms:newStr%natoms) = str2%ats(1:str2%natoms)

  do i=1,str2%natoms
     newStr%ats(i+str1%natoms)%r = sumavec(str2%ats(i)%r,point(x,y,z))
  enddo
  newStr%ats(1+str1%natoms:newStr%natoms)%resnum = str2%ats(1:str2%natoms)%resnum + str1%nres
  newStr%ats(1+str1%natoms:newStr%natoms)%molnum = str2%ats(1:str2%natoms)%molnum + str1%nmol
!
  newStr%distat2(1+str1%natoms:newStr%natoms,1+str1%natoms:newStr%natoms) = str2%distat2(1:str2%natoms,1:str2%natoms)
!
   do i=1,str1%natoms
   do j=1,str2%natoms
      newStr%distat2(i,j+str1%natoms) = calcDist2(str1%ats(i)%r,str2%ats(j)%r)
      newStr%distat2(j+str1%natoms,i) = newStr%distat2(i,j+str1%natoms)
   enddo
   enddo
!
  call allocateStructureResidues(newStr,str1%nres+str2%nres)
  newStr%res(1:str1%nres) = str1%res(1:str1%nres)
!
  newStr%res(1+str1%nres:newStr%nres) = str2%res(1:str2%nres)
!
  newStr%res(1+str1%nres:newStr%nres)%ini = str2%res(1:str2%nres)%ini + str1%natoms
  newStr%res(1+str1%nres:newStr%nres)%fin = str2%res(1:str2%nres)%fin + str1%natoms
  newStr%res(1+str1%nres:newStr%nres)%ica = str2%res(1:str2%nres)%ica + str1%natoms
  newStr%res(1+str1%nres:newStr%nres)%ico = str2%res(1:str2%nres)%ico + str1%natoms
  newStr%res(1+str1%nres:newStr%nres)%in  = str2%res(1:str2%nres)%in + str1%natoms
  newStr%res(1+str1%nres:newStr%nres)%molres = str2%res(1:str2%nres)%molres + str1%nmol
!
  call allocateStructureMols(newStr,str1%nmol+str2%nmol)
  newStr%mols(1:str1%nmol) = str1%mols(1:str1%nmol)  
!
  newStr%mols(1+str1%nmol:newStr%nmol)%ini = str2%mols(1:str2%nmol)%ini+str1%nres
  newStr%mols(1+str1%nmol:newStr%nmol)%fin = str2%mols(1:str2%nmol)%fin+str1%nres
! PENDENT determinar tipus de molecula 
  newStr%molType=COMPLEX
end function mergeStructures
!===============================================================================
subroutine markInterface (str, recNAts, dint0, dint1, irig, SetCoreCa, SetFrozen)
 type(struc),intent(INOUT) :: str
 real, intent(IN) :: dint0, dint1
 integer, intent(IN) :: setcoreca, setfrozen, recNAts, irig
 integer i,j
 real dint02,dint12
!
 dint02 = dint0**2
 dint12 = dint1**2
 str%ats%frozen=(setFrozen.eq.1)
 if (setcoreca.eq.1) str%ats%dummy = .true.
! if (setcoreca.eq.1) then
!    str%ats%dummy=.true.
!    do i=1,str%nres
!      str%ats(str%res(i)%ica)%dummy=.false.
!    enddo
! endif   
 do i=1,recNAts
 do j=recNats+1,str%natoms
!   write (0,*) i,j,str%distat2(i,j), dint12
   if (str%distat2(i,j).lt.dint12) then
      str%ats(i)%dummy=.false.
      str%ats(j)%dummy=.false.   
      if (str%distat2(i,j).lt.dint02) then
         str%ats(i)%frozen=.false.
         str%ats(j)%frozen=.false.
      endif
   endif      
 enddo
 enddo

 do i=1,str%natoms
    if (.not.str%ats(i)%dummy)then
       do j=1,str%natoms
          if(str%ats(j)%resNumPDB.eq.str%ats(i)%resNumPDB.and.str%ats(j)%molNum.eq.str%ats(i)%molNum)str%ats(j)%dummy=.false.
       enddo
    endif
    if (.not.str%ats(i)%frozen)then
      do j=1,str%natoms
         if(str%ats(j)%resNumPDB.eq.str%ats(i)%resNumPDB.and.str%ats(j)%molNum.eq.str%ats(i)%molNum)str%ats(j)%frozen=.false.
      enddo
   endif
   if(.not.str%ats(i)%dummy)then
      do j=str%natoms,1,-1
         if(str%ats(j)%resNumPDB.eq.str%ats(i)%resNumPDB.and.str%ats(j)%molNum.eq.str%ats(i)%molNum)str%ats(j)%dummy=.false.
      enddo
   endif
   if(.not.str%ats(i)%frozen)then
      do j=str%natoms,1,-1
         if(str%ats(j)%resNumPDB.eq.str%ats(i)%resNumPDB.and.str%ats(j)%molNum.eq.str%ats(i)%molNum)str%ats(j)%frozen=.false.
      enddo
   endif
 enddo

 do i=1,str%nres
   str%ats(str%res(i)%ica)%dummy=.false.
 enddo

 if(IRIG.eq.1)then
   do i=1,str%natoms
     if(str%ats(i)%atomId.eq.'CA')str%ats(i)%frozen=.true.
   enddo
 endif
end subroutine markInterface
!===============================================================================
 subroutine assignAtType (str,resLib)
 use resLibrary
 use utils
   type(struc), intent(INOUT) :: str
   type(residueLibrary), intent(IN) :: resLib
   type(atom) att
   integer i,j

   do i = 1,str%natoms
      att = getAtomFromLib(resLib,str%ats(i)%resIdbyAtom,str%ats(i)%atomId)
      if (att%atomId.ne.''.or.str%ats(i)%atomId.eq.'OXT ') then
         str%ats(i)%atType=att%atType
!         write (6,*) i,str%ats(i)%resIdbyAtom,str%ats(i)%atomId,att%atType
      else
         write (0,*) 'Unknown atom ', str%ats(i)%resIdbyAtom,str%ats(i)%resnum, str%ats(i)%atomId
         stop 1
      endif
   enddo
   if (str%moltype.eq.PROT) then
! N i C Terms Modificat per fragments multiples
     do i=1,str%nmol
        str%ats(str%res(str%mols(i)%ini)%in)%attype = getAtTypeFromLib(resLib,'NTE','NTE');
        do j=str%res(str%mols(i)%fin)%ini,str%res(str%mols(i)%fin)%fin
           if (eqId(str%ats(j)%atomId,'O').or.eqId(str%ats(j)%atomId,'OXT').or. &
                eqId(str%ats(j)%atomId,'C')) &
                str%ats(j)%atType =  getAtTypeFromLib(resLib, 'OXT', str%ats(j)%atomId)
        enddo
     enddo
    endif
 end subroutine assignAtType
!===============================================================================
 subroutine setBonds (str, resLib, tipCalc, recNatoms, cutoff)
 use resLibrary
   type(struc), intent(INOUT) :: str
   type(residueLibrary), intent(IN) :: resLib
   real, intent(IN) :: cutoff
   integer, intent(IN) :: tipCalc, recNatoms
   integer i,j

   call setIntraResBonds(str, resLib)
   if (tipCalc.eq.DOCKING) call setFrozenBonds(str, recNatoms, cutoff)
   if (str%moltype.eq.PROT.or.str%molType.eq.COMPLEX) then
      call setPeptideBonds(str, .false.) ! pendent fer crossbonds opcional
      call setSSBonds(str)
      call setHBonds(str)
   endif
   call prepBondList(str)
   do i=1,str%natoms-1
   do j=i+1,str%natoms
      str%bonds(j,i) = str%bonds(i,j)
   enddo
   enddo
 end subroutine setBonds
!===============================================================================
 subroutine setIntraResBonds (str, resLib)
 use resLibrary
 use utils
   type(struc), intent(INOUT) :: str
   type(residueLibrary), intent(IN) :: resLib
   type(residue) rr
   type(atom) at1, at2
   integer nr,i,j,k, oxtInd
!  logical eqId
!
   do nr=1,str%nres
      rr = getResidue(resLib,str%res(nr)%resId)
      do j=str%res(nr)%ini, str%res(nr)%fin-1
      do k=j+1, str%res(nr)%fin
         if (.not.eqId(str%ats(k)%atomId,'OXT')) then
           at1 = getAtomFromResidue(rr,str%ats(j)%atomId)
           at2 = getAtomFromResidue(rr,str%ats(k)%atomId)
           str%bonds(j,k) = rr%bonds(at1%ind,at2%ind)
         endif
      enddo
      enddo
   enddo
   if (str%moltype.eq.PROT.or.str%moltype.eq.COMPLEX) then
! Especial CTerm Modificat per fragments multiples
     do i=1,str%nmol
        oxtInd = findAtomInRes(str,str%mols(i)%fin,'OXT');
        if (oxtInd.gt.0) then
           str%bonds(findAtomInRes(str,str%mols(i)%fin,'C'),findAtomInRes(str,str%mols(i)%fin,'OXT')) = COV
           str%bonds(findAtomInRes(str,str%mols(i)%fin,'O'),findAtomInRes(str,str%mols(i)%fin,'OXT')) = COV
        endif   
     enddo
   endif
 end subroutine setIntraResBonds
!===============================================================================
subroutine setFrozenBonds(str,recNatoms, cutoff)
 type(struc), intent(INOUT) :: str
 integer, intent(IN) :: recNatoms
 real, intent(IN) :: cutoff
 real c2
 integer i,j
 c2=cutoff**2
 do i=1,recNatoms-1
 do j=i+1, recNatoms
   if (str%ats(i)%frozen.and.str%ats(j)%frozen.and.str%distat2(i,j).le.c2) str%bonds(i,j) = COV
 enddo
 enddo
 do i=recNatoms+1,str%natoms
 do j=i+1, str%natoms
   if (str%ats(i)%frozen.and.str%ats(j)%frozen.and.str%distat2(i,j).le.c2) str%bonds(i,j) = COV
 enddo
 enddo
end subroutine setFrozenBonds
!===============================================================================
 subroutine setPeptideBonds (str,crossb)
   type(struc), intent(INOUT) :: str
   integer nr
   logical crossb ! true pseudo peptide bonds between fragments
   do nr=1,str%nres-1
      if(crossb.or.str%res(nr)%molres.eq.str%res(nr+1)%molres) then
         str%bonds(str%res(nr)%ico,str%res(nr+1)%in)  = COV ! C-N
         str%bonds(str%res(nr)%ica,str%res(nr+1)%ica) = COV ! CA-(C)-(N)-CA
         str%bonds(str%res(nr)%ica,str%res(nr+1)%in)  = COV ! CA-(C)-N
         str%bonds(str%res(nr)%ico,str%res(nr+1)%ica) = COV ! C-(N)-CA
         str%bonds(findAtominRes(str,nr,'O'),str%res(nr+1)%in) = COV ! O-(C)-N
      endif
   enddo
  end subroutine setPeptideBonds
!===============================================================================
 subroutine setSSBonds (str)
 use utils
   type(struc), intent(INOUT) :: str
   integer nr1,nr2,sg1,sg2,cb1,cb2
   real :: rssmax2 = RSSMAX*2
! Disulfide bonds 
   do nr1=1, str%nres-1
      if (eqId(str%res(nr1)%resId,'CY_')) then
         sg1=findAtominRes(str, nr1, 'SG')
         cb1=findAtominRes(str, nr1, 'CB')
         do nr2=nr1+1,str%nres
            if (eqId(str%res(nr2)%resId,'CY_')) then
               sg2=findAtominRes(str, nr2, 'SG')
               cb2=findAtominRes(str, nr2, 'CB')
               if (sg1.ne.0.and.sg2.ne.0) then
                  if(str%distat2(sg1,sg2).le.rssmax2) &
                     str%bonds(sg1,sg2) = COV ! SG-SG
                     str%bonds(cb1,sg2) = COV ! CB-(SG)-SG
                     str%bonds(sg1,cb2) = COV ! SG-(SG)-CB
               endif
            endif         
         enddo
      endif
   enddo
 end subroutine setSSBonds
!===============================================================================
 subroutine setHBonds (str)
   type(struc), intent(INOUT) :: str
   integer nr1,nr2, atN, atO
   integer i
   integer lloc(1)
! hbonds main chain, el més proper possible es i, i+3 en hairpins posem i i+4 per compatibilitat
! Completem la matr simetria aqui per facilitar l'eliminacio de redundants
   do nr1 = 2,str%nres-4 ! comencem a 2 per que no podem mesurar una de les condicions
   do nr2 = nr1+4,str%nres
      atN = findAtominRes(str, nr1, 'N')
      atO = findAtominRes(str, nr2, 'O')
      if (checkHBNO(str,nr1,atN,nr2,atO)) then
         str%bonds(atN,atO)= HB
         str%bonds(atO,atN)= HB         
      endif
      atO=findAtomInRes(str,nr1,'O')
      atN=findAtomInRes(str,nr2,'N')
      if (checkHBNO(str,nr2,atN,nr1,atO)) then
         str%bonds(atO,atN)= HB
         str%bonds(atN,atO)= HB
      endif
   enddo
   enddo
! Mantenim un pont per residu, el de menor distancia, PENDENT mirar de no eliminar unics
   i=1
   do while (i.le.str%natoms)
      if (count(str%bonds(i,1:str%natoms).eq.2).gt.1) then
         lloc = minloc(str%distat2(i,1:str%natoms), (str%bonds(i,1:str%natoms).eq.2))
         where (str%bonds(i,1:str%natoms).eq.2)
            str%bonds(i,1:str%natoms)=0
         end where
         where (str%bonds(1:str%natoms,i).eq.2)
            str%bonds(1:str%natoms,i)=0
         end where
         str%bonds(i,lloc(1))=2
         str%bonds(lloc(1),i)=2
      endif
      i=i+1
   enddo
 end subroutine setHBonds 
!===============================================================================
 subroutine prepBondList(str)
   type(struc), intent(INOUT) :: str
   integer i,j

! anulem enllaços amb membres dummy
   do i=1,str%natoms
     if (str%ats(i)%dummy) then
       str%bonds(i,1:str%natoms) = 0
       str%bonds(1:str%natoms,i) = 0
     endif
   enddo      
!
   str%cov = allocatePairList(count(str%bonds.eq.COV))
   str%hbs = allocatePairList(count(str%bonds.eq.HB))
   str%cov%npairs=0
   str%hbs%npairs=0   
   str%other%npairs=0
   do i=1, str%natoms-1
   do j=i+1, str%natoms
      if (str%bonds(i,j).eq.COV) then
         str%cov%npairs = str%cov%npairs + 1
         str%cov%list(str%cov%npairs,1) = i
         str%cov%list(str%cov%npairs,2) = j
      elseif (str%bonds(i,j).eq.HB) then
         str%hbs%npairs = str%hbs%npairs + 1
         str%hbs%list(str%hbs%npairs,1) = i
         str%hbs%list(str%hbs%npairs,2) = j
      endif
   enddo
   enddo
 end subroutine prepBondList
 !===============================================================================
 function checkHBNO (str,resN,atN,resO,atO) result(hb)
 use utils
   logical hb
   type(struc), intent(IN) :: str
   integer, intent(IN) :: resN,resO,atN,atO
   real :: rnomax2 = RNOMAX**2
   real :: rnomin2 = RNOMIN**2
   real :: rncmax2 = RNCMAX**2
   real :: rncmin2 = RNCMIN**2
   real :: rcomax2 = RCOMAX**2
   real :: rcomin2 = RCOMIN**2
   hb = .false.
   if (resN.gt.1.and.resO.ne.0.and.atN.ne.0.and.atO.ne.0) then
      hb = (.not.eqId(str%res(resN)%resId,'PRO').and.&
         str%distat2(atN,atO).gt.rnomin2.and.str%distat2(atN,atO).lt.rnomax2.and.&
         str%distat2(str%res(resN-1)%ico,atO).gt.rcomin2.and.str%distat2(str%res(resN-1)%ico,atO).lt.rcomax2.and.&
         str%distat2(str%res(resN)%ica,atO).gt.rcomin2.and.str%distat2(str%res(resN)%ica,atO).lt.rcomax2.and.&
         str%distat2(atN,str%res(resO)%ico).gt.rncmin2.and.str%distat2(atN,str%res(resO)%ico).lt.rncmax2) 
   endif
 end function checkHBNO
 !===============================================================================
 subroutine calcSecStr (str) ! calculem SS amb el patro de ponts d'hidrogen
   type(struc), intent(INOUT) :: str
   integer, pointer :: nhel(:), nbet(:)
   integer, pointer :: hres(:)
   integer i, j, nhelx, nbets
   integer, parameter :: HELIX=1, BETA=2
   logical inh, inb
   allocate (hres(str%nres), nhel(str%nres), nbet(str%nres))
   hres = 0
   i=1

   do while (i.le.str%hbs%npairs)
      if ((str%ats(str%hbs%list(i,2))%resnum-str%ats(str%hbs%list(i,1))%resnum).eq.4) then
         j=i         
         do while (j.le.str%hbs%npairs.and.(str%ats(str%hbs%list(j,2))%resnum-str%ats(str%hbs%list(j,1))%resnum).eq.4.and. &
            j.le.str%hbs%list(i,2))
            hres(str%ats(str%hbs%list(j,1))%resnum) = HELIX
            j=j+1
         enddo
      else
         hres(str%ats(str%hbs%list(i,1))%resnum)=BETA
         hres(str%ats(str%hbs%list(i,2))%resnum)=BETA
      endif
      i=i+1
   enddo
! Recuperem forats d'1 residu Fer primer aixo per donar preferencia a completar en casos 0 1 0 1 0...
   do i=2,str%nres-1
      if (hres(i).eq.0.and.hres(i-1).eq.hres(i+1).and.hres(i-1).eq.1) then
          hres(i)=hres(i-1)
      endif
   enddo
! eliminem ss d'un residu
   do i=2,str%nres-1
      if (hres(i).ne.0.and.hres(i-1).eq.hres(i+1)) hres(i)=hres(i-1)
   enddo
!
   nhelx=0
   nbets=0
   nhel = 0
   nbet = 0
   inh =.false.
   inb =.false.
   do i=1,str%nres
      if (hres(i).eq.1) then
         if (.not.inh) then 
            nhelx = nhelx + 1
            inh=.true.
            inb=.false.
         endif
         nhel(i) = nhelx
      elseif (hres(i).eq.2) then
         if (.not.inb) then
            nbets = nbets + 1
            inb=.true.
            inh=.false.
         endif
         nbet(i) = nbets
      else
         inb=.false.
         inh=.false.
      endif
  enddo
  allocate(str%helix(nhelx))
  allocate(str%beta(nbets))
  str%helix = Fragment(99999,0)
  str%beta = Fragment(99999,0)
  str%nhelix=nhelx
  str%nbeta=nbets
  do i=1,str%nres
   if (nhel(i).gt.0) then
      str%helix(nhel(i))%ini = min(str%helix(nhel(i))%ini,i)
      str%helix(nhel(i))%fin = max(str%helix(nhel(i))%fin,i)
   elseif (nbet(i).gt.0) then
      str%beta(nbet(i))%ini = min(str%beta(nbet(i))%ini,i)
      str%beta(nbet(i))%fin = max(str%beta(nbet(i))%fin,i)
   endif
  enddo
 end subroutine calcSecStr
 !===============================================================================
 subroutine saveTopology (unt, str, ff, recNAts) 
 use potentials
  integer, intent(IN) :: unt, recNAts
  type(struc), intent(IN) :: str
  type(ffprm), intent(IN) :: ff
  type(atType) at
  integer i
  write (unt) topVersion
  write (unt) str%moltype
  write (unt) str%nmol, str%nres, str%natoms, recNAts
  write (unt) (str%mols(i),i=1,str%nmol)
  write (unt) (str%res(i)%molres,i=1,str%nres)
  if (str%moltype.eq.PROT.or.str%moltype.eq.COMPLEX)  &
     write (unt) (str%res(i)%in,str%res(i)%ica,str%res(i)%ico,i=1,str%nres)
  write (unt) (str%ats(i)%atomId,str%ats(i)%resnum,str%ats(i)%resIdByAtom,str%ats(i)%chainId, str%ats(i)%atType, &
                str%ats(i)%molnum, str%ats(i)%frozen,str%ats(i)%dummy,i=1,str%natoms)
  do i=1,str%natoms
    at =getPotential(ff,str%ats(i)%atType)
    write (unt) at%qq,at%gfree,at%vol,at%evdw,at%rvdw,at%rhc,at%mas
  enddo
  write (unt) str%cov%npairs, str%hbs%npairs, str%other%npairs
  write (unt) (str%cov%list(i,1),str%cov%list(i,2),&
      str%distat2(str%cov%list(i,1),str%cov%list(i,2)),i=1,str%cov%npairs)
  write (unt) (str%hbs%list(i,1),str%hbs%list(i,2),i=1,str%hbs%npairs)
  if (str%moltype.eq.PROT.or.str%moltype.eq.COMPLEX) then
      write (unt) str%nhelix
      write (unt) (str%helix(i)%ini,str%helix(i)%fin,i=1,str%nhelix)
      write (unt) str%nbeta
      write (unt) (str%beta(i)%ini,str%beta(i)%fin,i=1,str%nbeta)
  endif
 end subroutine saveTopology
!===============================================================================
 subroutine saveCoords (unt, str) 
  integer, intent(IN) :: unt
  type(struc), intent(IN) :: str
  integer i
  write (unt) str%natoms
  write (unt) (str%ats(i)%r,i=1,str%natoms)
 end subroutine saveCoords
!===============================================================================  
 function writeAtomId(at) result (txt)
 type(atomData) at
 character(len=15) txt
 write (txt,'(a4,1x,a1,i4,1x,a4)') at%resIdByAtom, at%chainId, at%resNumPDB, at%atomId
 end function writeAtomId
!===============================================================================  
 function writeResidueId(rr) result (txt)
 type(residueData) rr
 character(len=10) txt
 write (txt,'(a4,1x,a1,i4)') rr%resId, rr%chainId, rr%resNumPDB
 end function writeResidueId
!===============================================================================  
 END MODULE Structure
