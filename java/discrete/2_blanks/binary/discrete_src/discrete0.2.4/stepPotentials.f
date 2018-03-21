 MODULE stepPotentials
 
 integer, parameter :: MAXSTEPS = 3

 type stepPot
    real r,e
 end type stepPot

 type stepPotInt
    integer nstep, tipInt 
    type (stepPot) step(MAXSTEPS)
    logical active
 end type stepPotInt
 
 real, parameter :: FACTE = 4167. 
 real, parameter :: BOND=0, SS=1, COUL=2, HFIL=3, HFOB=4

 CONTAINS
!===============================================================================
 function getStepSSec (SIGMAGO, EGO, dist, active) result (st)
 type(stepPotInt) st
 real SIGMAGO,EGO,dist
 logical active
    st%nstep=2
    st%step(1)=stepPot((1.-SIGMAGO)*dist, -EGO*FACTE)
    st%step(2)=stepPot((1.+SIGMAGO)*dist, EGO*FACTE)
    st%active= active
    st%tipInt=SS
 end function getStepSSec
!===============================================================================
 function getStepCoul (rvdwij, DPSINT, DPSEXT, ecoul, HPS, active) result (st)
 type(stepPotInt) st
 real rvdwij, DPSINT, DPSEXT, ecoul, HPS
 logical active
    st%nstep=3
    st%step(1)=stepPot(rvdwij, -sign(3.,ecoul)*ecoul*FACTE)
    st%step(2)=stepPot(DPSINT, -(1.-HPS)*ecoul*FACTE)
    st%step(3)=stepPot(DPSEXT, -HPS*ecoul*FACTE)
    st%active=active
    st%tipInt=COUL
 end function getStepCoul
!===============================================================================
 function getStepHFil (rvdwij, DHF, esolv, active) result (st)
 type(stepPotInt) st
 real rvdwij, DHF, esolv
 logical active
    st%nstep=2
    st%step(1)=stepPot(rvdwij, -1.5*esolv*FACTE)
    st%step(2)=stepPot(DHF, -esolv*FACTE)
    st%active=active
    st%tipInt=HFIL
 end function getStepHFil
!===============================================================================
 function getStepHFob (rvdwij, DHF, esolv, active) result (st)
 type(stepPotInt) st
 real rvdwij, DHF, esolv
 logical active
    st%nstep=2
    st%step(1)=stepPot(rvdwij, 3.0*esolv*FACTE)
    st%step(2)=stepPot(DHF, -esolv*FACTE)
    st%active=active
    st%tipInt=HFOB
 end function getStepHFob
!===============================================================================
 END MODULE stepPotentials
