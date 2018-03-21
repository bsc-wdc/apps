!
! Authors : Josep Ll. Gelpi, Agusti Emperador
! Subject : Main Setup program for DMD
! Revision: $id$
!
! v. 0.3. Prepared for multiple molecules separated by TER, possible coverage of EDMD, specific for non-protein
!
program ligandSetup
use commLine
use Structure
use potentials
use resLibrary
!
   integer, parameter :: NFILES = 6
   integer unit_i, unit_o
!
   type(commLineOption) :: files(NFILES) = (/& ! COMPTE: cal que -o sigui el 6 per evitar problemes en debug
      commLineOption("-rlib","reslib","formatted","old","Residue Library"),&
      commLineOption("-pot","potential","formatted","old","DMD Potentials"),&
      commLineOption("-pdbin","pdbin","formatted","old","Input structure(PDB)"),&
      commLineOption("-top","topology","unformatted","unknown","Topology"),&
      commLineOption("-r", "coordinates","unformatted","unknown","Coordinates"), &
      commLineOption("-o","log","formatted","unknown","Log File") /)
!
   type(residueLibrary) :: resLib
   type(ffprm) :: ff
   type(struc) :: str
!
   type(residue) r
   integer i,j,k,l
   type(atom) att
!
   call readParameters (files, unit_i, unit_o, NFILES)
   call printFileSummary(files, unit_o)
   write (unit_o,*)
   resLib = loadResLibrary(openFn(files, '-rlib'))
   write(unit_o,'(1X, "Residue Library: " i4 " residues / ", i5, " atoms loaded")') resLib%nres, sum(resLib%residues%natoms)
   ff = loadPotentials(openFn(files, '-pot'))
   write(unit_o,'(1X, "Atom Types:      ", i4, " types")') ff%ntypes
   str = loadStructurePDB(openFn(files, '-pdbin'))
!
   str%molType=SMALL
!
   write(unit_o,'(1X, "Structure:       ", i4, " molecules / ", i4, " residues / ", i5, " atoms loaded")') &
         str%nmol, str%nres, str%natoms
   write(unit_o, '(1X,3i5)') (i, str%mols(i),i=1,str%nmol)
!
   call assignAtType(str,resLib)
   call setBonds(str, resLib)
!
   call saveTopology(openFn(files,'-top'),str,ff)
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
!
 end subroutine readParameters
!======================================================================
 subroutine header (unit_o)
  integer, intent(IN) :: unit_o
   write (unit_o, *) "================================================="
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "=                DMDSetup  (v. 0.2)             ="
   write (unit_o, *) "=                   (c) 2011                    ="
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "================================================="
   write (unit_o, *)
 end subroutine header
!======================================================================

