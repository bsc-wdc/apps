!
! $Date: 2011-09-16 09:39:13 $
! $Id: commLine.f,v 1.4 2011-09-16 09:39:13 gelpi Exp $
! $Revision: 1.4 $
!
MODULE commLine

 type commlineOption
    character(len=10) descr
    character(len=100) filename
    character(len=20) fmt, status
    character(len=50) help
 end type commlineOption
 
 
CONTAINS
 
function openFn (opts, descr, unit, ext) result(unitf)
 type (commLineOption) opts(:)
 integer unitf, err
 character*(*), intent(IN), optional :: descr
 character*(*), optional :: ext
 integer, optional :: unit
 if (.not.present(ext)) ext=''
 if (.not.present(unit)) then
    unitf = findDescr(opts,descr)
 else
    unitf = unit
 end if
 if (unitf.gt.0)  then
    open (unitf, file=trim(opts(unitf)%filename)//trim(ext), status=opts(unitf)%status, form=opts(unitf)%fmt, iostat=err)
    if (err.ne.0) then
       write (0, '("Error opening file: ", a50)') opts(unitf)%filename
       stop 1
    end if
 endif
end function openFn      

function findDescr (opts, descr) result(unit)
 integer unit
 character*(*), intent(IN) :: descr
 type(commLineOption), intent(IN) :: opts(:)
 integer i
 i=1
 do while (trim(descr).ne.opts(i)%descr.and.i.lt.ubound(opts,1))
   i=i+1
 end do
 if (trim(descr).eq.trim(opts(i)%descr)) then
    unit=i
 else
    unit=0
 end if
end function findDescr

subroutine inputArgs(opts)
 type(commLineOption), intent(INOUT) :: opts(:)
 integer iarg, unit
 character(len=80) arg
 iarg=1
 do while (iarg.le.iargc())
    call getarg(iarg,arg)
    if (trim(arg).eq.'-h'.or.trim(arg).eq.'--help'.or.trim(arg).eq.'--usage') then
      call writeHelpText(opts)
      stop 
    end if
    unit = findDescr(opts,arg)
    if (unit.eq.0) then
       write (0,'("Error: unknown option (",A5,")")') arg
       call writeHelpText(opts)  
       stop 1
    else
       iarg=iarg+1
      call getarg(iarg, opts(unit)%filename)
      if (len(opts(unit)%filename).le.0) then
         write (0, '("Error: missing parameter (",a10,"). Type ""DMDSetup -h"" for help")')
         stop 1
      end if
    end if
    iarg=iarg+1
 end do
end subroutine inputArgs

subroutine writeHelpText (opts)
 type(commLineOption), intent(in) :: opts(:)
 integer i
 write (0,*) "Usage:"
 write (0,'(2X,A10,A10,A50)') (opts(i)%descr, opts(i)%filename, opts(i)%help, i=1,ubound(opts,1))
 write (0,*)
end subroutine writeHelpText
 
function fileName(opts, descr, unit) result(fn)
 character(len=50) fn
 character*(*), intent(in), optional :: descr
 integer, optional :: unit
 integer unitf
 type(commLineOption), intent(IN) :: opts(:)
 if (present(unit)) then
    unitf = unit
 else
    unitf = findDescr(opts,descr)
 end if
 if (unitf.gt.0) then
    fn = opts(unitf)%filename
 else
    fn = "Unknown"
 end if
end function fileName

subroutine printFileSummary(opts, unit)
 type (commLineOption), intent(IN) :: opts(:)
 integer unit, i
 write (unit,'( 2x, "I N P U T  F I L E S")')
 write (unit,'( 2X, "====================")')
 write (unit, '(2X, A20,": ",A)') (opts(i)%help,trim (opts(i)%filename), i=1,ubound(opts,1))
end subroutine printFileSummary

end module commLine
 
