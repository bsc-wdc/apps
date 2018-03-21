!
! Authors : Josep Ll. Gelpi, Agusti Emperador
! Subject : DMD Residue Library module
! Revision: $id$
!
MODULE resLibrary

 integer, parameter :: MAXRES = 50, MAXATPERRES = 30, COV=1

 type atom
   character(LEN=4) :: atomId
   character(LEN=4) :: atType
   character(LEN=1) :: element
   integer resNum
   integer ind
 end type atom

 type residue
   character(LEN=4) :: resId
   character(LEN=30) :: resName
   type (atom) atoms(MAXATPERRES)
   integer :: bonds (MAXATPERRES, MAXATPERRES)
   integer nAtoms
   integer ind
 end type residue

 type residueLibrary
   type (residue) :: residues(MAXRES)
   integer nRes
 end type residueLibrary

CONTAINS
!================================================================================
 function emptyResidue () result (rr)
   type(residue) rr
   rr%resId=''
   rr%resName=''
   rr%atoms = atom('','','',0,0)
   rr%bonds = 0
 end function emptyResidue
!================================================================================
 function getResidue (rl, resId) result (rr)
 use utils
  type(residue) rr
  type(residueLibrary) rl
  character(LEN=*) resId
  integer i
  
  rr=emptyResidue()
  i=1
  do while (i.lt.rl%nRes.and..not.eqId(rl%residues(i)%resId,resId))
   i=i+1
  end do
  if (eqId(rl%residues(i)%resId,resId)) then
   rr = rl%residues(i)
  endif
  
 end function getResidue
!===============================================================================
 function getAtomFromResidue (rres,atomId) result(at)
 use utils
  type(atom) at
  type(residue), intent(IN) :: rres
  character(LEN=*) atomId
  integer i
  
  i=1
  do while (i.le.rres%nAtoms.and..not.eqId(rres%atoms(i)%atomId,atomId)) 
   i=i+1
  enddo
  if (eqId(rres%atoms(i)%atomId,atomId)) then
   at = rres%atoms(i)
  else
   at = atom('','','',0,0)
  endif
  
 end function getAtomFromResidue
!================================================================================
 function getAtomFromLib (rl, resId, atomId) result(at)
 use utils
  type(atom) at
  type(residue) rr
  type(residueLibrary) rl
  character(len=*) resId,atomId
  rr = getResidue(rl, resId)
  if (eqId(rr%resId,resId)) then
   at = getAtomFromResidue(rr,atomId)
  else
   at = atom('','','',0,0)
  endif
 end function getAtomFromLib
!================================================================================
 function getAtTypeFromLib (rl, resId, atomId) result(ty)
  type(residueLibrary) rl
  type(atom) at
  character(*) resId, atomId
  character(len=4) ty
  at = getAtomFromLib(rl, resId, atomId)
  ty = at%atType
 end function getAtTypeFromLib
!================================================================================
 function loadResLibrary (unt) result (rlib)
   USE utils
   type(residueLibrary) :: rlib
   integer unt, i, j, k, kk, line, nargs
   logical ok
   character(len=80) str
   character(len=50) bondStr(MAXRES, MAXATPERRES)
   character(len=20) args(MAXATPERRES + 3)
!
   i = 0
   line = 1
10 read (unt, '(A80)', end=20) str
   if (str(1:1).eq.' '.or.str(1:1).eq.'#') goto 10
   call parse(str, ' ',args,nargs)
   if (args(1).eq.'RESIDUE') then
      i = i + 1
      rlib%nRes = i
      rlib%residues(i)%resId = trim(args(2))
      rlib%residues(i)%resName  = args(3)
      rlib%residues(i)%nAtoms = 0
      rlib%residues(i)%ind = i
   else
      if (trim(args(1)).eq.rlib%residues(i)%resId) then 
         j = rlib%residues(i)%nAtoms + 1
         rlib%residues(i)%nAtoms = j
         rlib%residues(i)%atoms(j)%ind = j
         rlib%residues(i)%atoms(j)%atomId = args(2)
         rlib%residues(i)%atoms(j)%atType = args(3)
         rlib%residues(i)%atoms(j)%element = rlib%residues(i)%atoms(j)%atomId(1:1)
         bondStr(i,j) = str(13:)
         call compact(bondStr(i,j))
      else
         write (0,'("Error reading residue library at line ", i4, 1X,"(", A,")")') line, trim(str)
         stop 1
      endif
   endif
   line=line+1
   goto 10
20 continue
! gestio bonds
   do i = 1,rlib%nRes
      rlib%residues(i)%bonds = 0
      ok=.false.
      do j = 1,rlib%residues(i)%nAtoms
         call parse(bondStr(i,j),' ', args, nargs)
         do k = 1,nargs
            if (trim(args(k)).ne.'') then 
               do kk=1,rlib%residues(i)%natoms
                  if (eqId(rlib%residues(i)%atoms(kk)%atomId,trim(args(k)))) then
                     rlib%residues(i)%bonds(j,kk) = COV
                     rlib%residues(i)%bonds(kk,j) = COV      
                     ok=.true.
                  endif
               enddo
               if (.not.ok) then
                  write (0,*) "Error reading connectivity, atom not found ", trim(args(k)),&
                   " on residue ", rlib%residues(i)%resId
                  stop 1
               endif
            endif
         enddo
      enddo
   enddo

 end function loadResLibrary
!===============================================================================================
 subroutine saveResLibrary (unt, rlib)
  type(residueLibrary), intent(IN) :: rlib
  integer, intent(IN) :: unt
  integer i,j
  write (unt,'(1X,i5)') rlib%nres
  write (unt,'(1X,a4,a20,i2)') (rlib%residues(i),i=1,rlib%nres)
  do i=1,rlib%nRes
     write (unt,*) (rlib%residues(i)%atoms(j), j=1,rlib%residues(i)%nAtoms)
     write (unt,*) rlib%residues(i)%bonds
  enddo
 end subroutine saveResLibrary
!===============================================================================================
 

END MODULE resLibrary
