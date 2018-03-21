 !===============================================================================   
 subroutine readCommLine (files, unit_i, unit_o, NFILES)
 USE commLine
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
 end subroutine readCommLine
!===============================================================================   
 subroutine programHeader(unit_o)
   integer unit_o
#ifdef OMP
   integer omp_get_max_threads
#endif
   write (unit_o, *) "================================================="
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "=               DISCRETE  (0.2.4a)              ="
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "=     A. Emperador, J. L. Gelpi, M.Orozco       =" 
   write (unit_o, *) "=                                               ="
   write (unit_o, *) "=                  (c) 2011                     =" 
   write (unit_o, *) "================================================="
   write (unit_o, *)
#ifdef OMP
   write (unit_o, '(" Running OpenMP version on ", i3, " processors")') omp_get_max_threads()
   write (unit_o, *)
#endif
 end subroutine programHeader
!===============================================================================   
 subroutine errorAllocmem (ioerr, text)
  integer ioerr
  character(len=*) text
  if (ioerr.ne.0) then
    write (0, '("Error allocating memory",a30)') text
    stop 1
  endif
 end subroutine errorAllocmem
!===============================================================================   
 subroutine writeSnapshot (unit_traj, ibloc, r, atom, res, chain, rnum, natom, oid, onatom, OUTFORMAT)
 use geometryDP
  integer, intent(IN) :: natom, unit_traj, ibloc, onatom, OUTFORMAT
  type(pointDP), intent(IN) :: r(natom)
  character(len=4), intent(IN) :: atom(onatom), res(onatom)
  character(len=1), intent(IN) :: chain(onatom)
  integer, intent(IN) :: rnum(onatom), oid(onatom)
  integer i, j
  if (OUTFORMAT.eq.1) then
   write (unit_traj,'(10f8.3)') (r(i),i=1,natom)
  else  
     write (unit_traj, '("MODEL",8X,I4)') ibloc
     do i = 1,natom
      j=oid(i)
      write (unit_traj, '("ATOM",2X,I5,2X,A3,1X,A3,1X,A1,1X,I3,4X,3F8.3)') i, atom(j), res(j), chain(j), rnum(j), r(i)
     enddo
     write (unit_traj, '("ENDMDL")')
  endif
 end subroutine writeSnapshot
!===============================================================================   

