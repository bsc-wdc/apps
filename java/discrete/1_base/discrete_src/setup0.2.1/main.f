!
! Authors : Josep Ll. Gelpi, Agusti Emperador
! Subject : Main Setup program for DMD
! Revision: $id$
!
! v. 0.2. Prepared for multiple molecules separated by TER, possible coverage of EDMD
! v, 0.2.1. Prepared for protein-protein docking. implementes multiscale en frozen atoms
!
program DMDSetup
use commLine
use Structure
use potentials
use resLibrary
!
   integer, parameter :: NFILES = 8
   integer unit_i, unit_o
!
   type(commLineOption) :: files(NFILES) = (/& ! COMPTE: cal que -o sigui el 6 per evitar problemes en debug
      commLineOption("-rlib","reslib","formatted","old","Residue Library"),&
      commLineOption("-pot","potential","formatted","old","DMD Potentials"),&
      commLineOption("-pdbin","pdbin","formatted","old","Input structure(PDB)"),&
      commLineOption("-top","topology","unformatted","unknown","Topology"),&
      commLineOption("-r", "coordinates","unformatted","unknown","Coordinates"), &
      commLineOption("-o","log","formatted","unknown","Log File") , &
      commLineOption("-ligpdbin", "ligpdbin","formatted","old","Input ligand structure(PDB)"), &
      commLineOption("-i", "null","formatted","unknown","Settings") &
      /)
!
   type(residueLibrary) :: resLib
   type(ffprm) :: ff
   type(struc) :: recstr, ligstr, str
!
   integer i,k,l
! 
   integer tipcalc, recNatoms, setcoreca, setfrozen, irig
   real dint0, dint1, offsetx, offsety,offsetz, bondcutoff
!
   namelist /setup/tipcalc, dint0, dint1, setfrozen, setcoreca, &
      offsetx,offsety,offsetz, irig, bondcutoff
   tipcalc = MD
   dint0 = 8.
   dint1 = 12.
   setfrozen=1
   bondcutoff=8.
   setcoreca=1
   offsetx=0.
   offsety=0.
   offsetz=0.
!
   call readParameters (files, unit_i, unit_o, NFILES)
   call printFileSummary(files, unit_o)
   write (unit_o,*)
!
   write (6,*) files(8)%filename
   if (trim(files(8)%filename).ne.'null') read(openFn(files,'-i'), setup)
!      
   resLib = loadResLibrary(openFn(files, '-rlib'))
   write(unit_o,'(1X, "Residue Library: " i4 " residues / ", i5, " atoms loaded")') resLib%nres, sum(resLib%residues%natoms)
   ff = loadPotentials(openFn(files, '-pot'))
   write(unit_o,'(1X, "Atom Types:      ", i4, " types")') ff%ntypes
   write(unit_o,*)
   if (TIPCALC.eq.MD) then
      str = loadStructurePDB(openFn(files, '-pdbin'))
      write(unit_o,'(1X, "Structure:       ", i4, " molecules / ", i4, " residues / ", i5, " atoms loaded")') &
            str%nmol, str%nres, str%natoms
      write(unit_o, '(" Molecule ",i3,": ",2i5)') (i, str%mols(i),i=1,str%nmol)
      recNatoms=str%natoms
   elseif (TIPCALC.eq.DOCKING) then
      write(unit_o,'(1X,"Setup for Protein-protein docking")') 
      recstr = loadStructurePDB(openFn(files, '-pdbin'))
      write(unit_o,'(1X, "Receptor Structure:    ", i4, " molecules / ", i4, " residues / ", i5, " atoms loaded")') &
      recstr%nmol, recstr%nres, recstr%natoms
      write(unit_o, '(" Receptor Molecule",i3,": ",2i5)') (i, recstr%mols(i),i=1,recstr%nmol)
      recNatoms=recStr%natoms
!
      ligstr = loadStructurePDB(openFn(files, '-ligpdbin'))
      write(unit_o,'(1X, "Ligand Structure:      ", i4, " molecules / ", i4, " residues / ", i5, " atoms loaded")') &
         ligstr%nmol, ligstr%nres, ligstr%natoms
      write(unit_o, '(" Ligand Molecule  ",i3,": ",2i5)') (i, ligstr%mols(i),i=1,ligstr%nmol)
      write(unit_o,'(" Building complex...")')         
      str = mergeStructures (recstr, ligstr, offsetx, offsety,offsetx)         
      write(unit_o,'(1X, "Complex Structure:     ", i4, " molecules / ", i4, " residues / ", i5, " atoms loaded")') &
         str%nmol, str%nres, str%natoms
      write(unit_o, '(" Complex Molecule  ",i3,": ",2i5)') (i, str%mols(i),i=1,str%nmol)
      write(unit_o,*)
!
      call markInterface(str, recNatoms, dint0, dint1, irig, SetCoreCa, SetFrozen)
      write (unit_o,'(1X,"Getting interface...")')
      write (unit_o,'(1X,"Interface dist. ", f5.1," All Atom layer: ",f5.1)') dint0, dint1
      write (unit_o,'(1X,"           Receptor    Ligand")')
      write (unit_o,'(1X,"Interface  ",         i7,"    ", i7)') count(.not.str%ats(1:recStr%natoms)%frozen), &
         count(.not.str%ats(recStr%natoms+1:str%natoms)%frozen)
      write (unit_o,'(1X,"Frozen     ",         i7,"    ", i7)') & 
         count(str%ats(1:recStr%natoms)%frozen)-count(str%ats(1:recStr%natoms)%dummy), &
         count(str%ats(recStr%natoms+1:str%natoms)%frozen)-count(str%ats(recStr%natoms+1:str%natoms)%dummy) 
      write (unit_o,'(1X,"Discarded  ",         i7,"    ", i7)') count(str%ats(1:recStr%natoms)%dummy), &
         count(str%ats(recStr%natoms+1:str%natoms)%dummy)
      write (unit_o, *)         
   else
      write(0,'(" Calculation type not implemented")')
      stop 1
   endif
!            
   call assignAtType(str,resLib)
   call setBonds(str, resLib, tipCalc, recNatoms, bondCutoff)
!
   write (unit_o,*) 'Hydrogen bonds' 
   do i=1,str%hbs%npairs
      k = str%hbs%list(i,1)
      l = str%hbs%list(i,2)
      write (unit_o,'(2X,i4," ",a15,": ",i4," ",a15,f8.3)')  & 
         str%ats(k)%molnum, writeAtomId(str%ats(k)), & 
         str%ats(l)%molnum, writeAtomId(str%ats(l)), &
         sqrt(str%distat2(k,l))
   enddo
!   
   call calcSecStr(str)
   write (unit_o, *) str%nhelix, " Alfa Helices"
   do i=1, str%nhelix
      write (unit_o,'(1x,2i3," ",a10," - ",i3, " ", a10)') i, &
         str%res(str%helix(i)%ini)%molres, writeResidueId(str%res(str%helix(i)%ini)), &
         str%res(str%helix(i)%fin)%molres, writeResidueId(str%res(str%helix(i)%fin))
   enddo
   write (unit_o, *) str%nbeta, " Beta strands"
   do i=1, str%nbeta
      write (unit_o,'(1x,2i3," ",a10," - ",i3, " ",a10)') i,  &
         str%res(str%beta(i)%ini)%molres, writeResidueId(str%res(str%beta(i)%ini)), & 
         str%res(str%beta(i)%fin)%molres, writeResidueId(str%res(str%beta(i)%fin))
   enddo
!
   call saveTopology(openFn(files,'-top'),str,ff,recNatoms)
   call saveCoords(openFn(files,'-r'),str)
   write (unit_o,*) "Topology & Coodinates files saved"
 end
!======================================================================
 subroutine readParameters (files, unit_i, unit_o, NFILES)
 use commLine
!
 integer, intent(IN) :: NFILES
 integer, intent(OUT) :: unit_i, unit_o
 type(commLineOption), intent(INOUT) :: files(NFILES) 
!
 call inputArgs(files)
 unit_i = openFn(files, '-i')
 if (fileName(files,'-o').ne.'log') then
   unit_o = openFn(files, '-o')
 else
   unit_o = 6
 endif
 call header (unit_o)
 end subroutine readParameters
!======================================================================
 subroutine header (unit_o)
  integer, intent(IN) :: unit_o
   write (unit_o, *) "================================================="
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "=                DMDSetup  (v. 0.2.1)           ="
   write (unit_o, *) "=                   (c) 2011                    ="
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "================================================="
   write (unit_o, *)
 end subroutine header
!======================================================================
