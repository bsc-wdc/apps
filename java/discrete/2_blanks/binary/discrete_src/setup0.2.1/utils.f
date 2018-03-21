!
! Authors : Josep Ll. Gelpi, Agusti Emperador
! Subject : Miscelania utils
! Revision: $id$
!
 MODULE utils

 CONTAINS

!================================================================================
 function eqId(a,b) result (eq)
   character(*) a,b
   logical eq
   eq = (a.eq.b.or.(a(1:2).eq.b(1:2).and.(a(3:3).eq.'_'.or.b(3:3).eq.'_')))
 end function eqId
!================================================================================

 subroutine parse(str,delims,args,nargs)
! Parses the string 'str' into arguments args(1), ..., args(nargs) based on
! the delimiters contained in the string 'delims'. Preceding a delimiter in
! 'str' by a backslash (\) makes this particular instance not a delimiter.
! The integer output variable nargs contains the number of arguments found.
  integer nargs, na, i, lenstr, k
  character(len=*) :: str,delims
  character(len=len_trim(str)) :: strsav
  character(len=*),dimension(:) :: args
  strsav=str
  call compact(str)
  na=size(args)
  do i=1,na 
   args(i)=' '
  end do  
  nargs=0
  lenstr=len_trim(str)
  if(lenstr==0) return
  k=0
  do
    if(len_trim(str) == 0) exit
    nargs=nargs+1
    call split(str,delims,args(nargs))
    call removebksl(args(nargs))
  end do   
  str=strsav
 end subroutine parse
 
 subroutine compact(str)
! Converts multiple spaces and tabs to single spaces; deletes control characters;
! removes initial spaces.
  integer lenstr, isp, k, i, ich
  character(len=*):: str
  character(len=1):: ch
  character(len=len_trim(str)):: outstr 
  str=adjustl(str)
  lenstr=len_trim(str)
  outstr=' '
  isp=0
  k=0 
  do i=1,lenstr
   ch=str(i:i)
   ich=iachar(ch)   
   select case(ich)
     case(9,32)     ! space or tab character
       if(isp==0) then
         k=k+1
         outstr(k:k)=' '
       end if
       isp=1       
     case(33:)      ! not a space, quote, or control character
       k=k+1
       outstr(k:k)=ch
       isp=0       
  end select
 end do
 str=adjustl(outstr)
 end subroutine compact

!**********************************************************************

subroutine removesp(str)

! Removes spaces, tabs, and control characters in string str
 integer lenstr, i, ich, k
 character(len=*):: str
 character(len=1):: ch
 character(len=len_trim(str))::outstr

 str=adjustl(str)
 lenstr=len_trim(str)
 outstr=' '
 k=0

 do i=1,lenstr
  ch=str(i:i)
  ich=iachar(ch)
  select case(ich)    
    case(0:32)  ! space, tab, or control character
         cycle       
    case(33:)  
      k=k+1
      outstr(k:k)=ch
  end select
end do

str=adjustl(outstr)

end subroutine removesp

subroutine split(str,delims,before,sep)

! Routine finds the first instance of a character from 'delims' in the
! the string 'str'. The characters before the found delimiter are
! output in 'before'. The characters after the found delimiter are
! output in 'str'. The optional output character 'sep' contains the 
! found delimiter. A delimiter in 'str' is treated like an ordinary 
! character if it is preceded by a backslash (\). If the backslash 
! character is desired in 'str', then precede it with another backslash.
 integer i, lenstr, k, ibsl, ipos, iposa
 character(len=*) :: str,delims,before
 character,optional :: sep
 logical :: pres
 character :: ch,cha

pres=present(sep)
str=adjustl(str)
call compact(str)
lenstr=len_trim(str)
if(lenstr == 0) return        ! string str is empty
k=0
ibsl=0                        ! backslash initially inactive
before=' '
do i=1,lenstr
   ch=str(i:i)
   if(ibsl == 1) then          ! backslash active
      k=k+1
      before(k:k)=ch
      ibsl=0
      cycle
   end if
   if(ch == '\') then          ! backslash with backslash inactive
      k=k+1
      before(k:k)=ch
      ibsl=1
      cycle
   end if
   ipos=index(delims,ch)         
   if(ipos == 0) then          ! character is not a delimiter
      k=k+1
      before(k:k)=ch
      cycle
   end if
   if(ch /= ' ') then          ! character is a delimiter that is not a space
      str=str(i+1:)
      if(pres) sep=ch
      exit
   end if
   cha=str(i+1:i+1)            ! character is a space delimiter
   iposa=index(delims,cha)
   if(iposa > 0) then          ! next character is a delimiter
      str=str(i+2:)
      if(pres) sep=cha
      exit
   else
      str=str(i+1:)
      if(pres) sep=ch
      exit
   end if
end do
if(i >= lenstr) str=''
str=adjustl(str)              ! remove initial spaces
return

end subroutine split

!**********************************************************************

subroutine removebksl(str)

! Removes backslash (\) characters. Double backslashes (\\) are replaced
! by a single backslash.
 integer lenstr, i, k, ibsl
 character(len=*):: str
 character(len=1):: ch
 character(len=len_trim(str))::outstr

str=adjustl(str)
lenstr=len_trim(str)
outstr=' '
k=0
ibsl=0                        ! backslash initially inactive

do i=1,lenstr
  ch=str(i:i)
  if(ibsl == 1) then          ! backslash active
   k=k+1
   outstr(k:k)=ch
   ibsl=0
   cycle
  end if
  if(ch == '\') then          ! backslash with backslash inactive
   ibsl=1
   cycle
  end if
  k=k+1
  outstr(k:k)=ch              ! non-backslash with backslash inactive
end do

str=adjustl(outstr)

end subroutine removebksl

!**********************************************************************


 END MODULE utils
