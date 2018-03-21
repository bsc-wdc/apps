 MODULE paramSet
! Input param
!
   real, save :: &
      fvdw = 0.6, &
         fsolv = 0.5, &
         eps =  1., &
         dpsext = 7.0, &
         dpsint = 5.0, &
         dhf = 5.0, &
         dcut = 2.0,&
         svdw = 0., &
         shc = 0., &
         sigma = 0.01, &
         temp = 300., &
         rcutgo = 10., &
         sigmago = 0.1, &
         tmin = 1.e-22, &
         ego = 1., &
         hps = 0.4, &
         rnomax=4.1, &
         rnomin=2.5, &
         rcomax=5., &
         rcomin=3.2, &
         rcutcoul2=0., &
         rcutsolv2=0., &
         xbeta=1000., &
         sclim=0.1
     integer, save :: &
        isolv = 1, &
        nbloc = 1000, &
        idab = 1, &
        igoab = 1, &
        iwr = 1, &
        seed = 2381, &
        TCALC = 1, & ! E Const 0,  T Const 1, TBany 2
        outformat = 0, &
        idims = 0, &
        rst=0, &
        rstv=0
      real*8, save :: &
        tsnap = 1000., &
        tcorr = 100., &
        tact = 50., &
        trect = 100., &
        tini = 0.

CONTAINS
!===============================================================================
 subroutine readInputParamSet (unit_i)
   integer, intent(IN) :: unit_i
!   
   namelist /input/ tsnap,tcorr,tact,sigma,temp,seed,&
            nbloc,rcutgo,sigmago,tmin,dhf,dcut,&
            ego,isolv,fsolv,fvdw,eps,hps,dpsint,dpsext,idab,igoab,iwr,svdw,shc,&  
            rnomax,rnomin,rcomax,rcomin, tcalc, outformat, trect, xbeta, sclim, idims, &
            rst, tini, rstv
!
   read (unit_i, INPUT)
   ! checking 
   if (TRECT.gt.TSNAP) TRECT = TSNAP
   if (TCORR.gt.TRECT) TCORR = TRECT
   if (TACT.gt.TCORR)  TACT = TCORR
   rcutcoul2 = (DPSEXT + DCUT)**2
   rcutsolv2 = (DHF + DCUT)**2
   if (RST.eq.1) RSTV = 1
 end subroutine readInputParamSet
!===============================================================================
 subroutine writeInputParamSet (unit_o)
   integer, intent(IN) :: unit_o
   ! Copiem l'arxiu de parametres. Pendent format
   write (unit_o, *)
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | CALCULATION PARAMETERS                                   |")')
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | Simulation settings                                      |")')
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | Simulation Time (ps) (Nbloc x TSnap)       |",f12.3," |")') NBLOC * TSNAP / 1.e3
   write (unit_o, '(" | Output structure (fs)             | TSnap  |",f12.3," |")') TSNAP 
   if (IDIMS.eq.1) &
   write (unit_o, '(" | Re-scoring target (fs)            | Trect  |",f12.3," |")') TRECT
   write (unit_o, '(" | Update velocities (fs)            | Tcorr  |",f12.3," |")') TCORR
   write (unit_o, '(" | Update Lists, collision times (fs)| Tact   |",f12.3," |")') TACT
   write (unit_o, '(" | Min. accepted colision time (fs)  | TMin   |",f12.8," |")') TMIN*1.e15   
   write (unit_o, '(" | Temperature (K)                   | Temp   |",6X,f6.2, " |")') TEMP
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | Well Potential Definitions                               |")')  
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | Cov. Bond relative well width     | Sigma  |",7X,f5.2, " |")') SIGMA
   write (unit_o, '(" | N-O (min-max) (Angs)              | rno    |",f5.2,X,"-",f5.2, " |")') rnomin, rnomax
   write (unit_o, '(" | C-O (min-max) (Angs)              | rco    |",f5.2,X,"-",f5.2, " |")') rcomin, rcomin
   write (unit_o, '(" | CutOff Beta restrains (A)         | RCutGo |",7X,f5.2, " |")') RCUTGO
   write (unit_o, '(" | SSec relative well width          | SigmaGo|",7X,f5.2, " |")') SIGMAGO
   write (unit_o, '(" | SSec wells depth                  | EGo    |",7X,f5.2, " |")') EGO
   write (unit_o, '(" | Coulombic Int. wall (A)           | DPSInt |",7X,f5.2, " |")') DPSINT
   write (unit_o, '(" | Coulombic Ext. wall (A)           | DPSExt |",7X,f5.2, " |")') DPSEXT
   write (unit_o, '(" | 1/Dielectric                      | Eps    |",7X,f5.2, " |")') EPS
   write (unit_o, '(" | E. Fraction on 2nd Coul. well     | HPS    |",7X,f5.2, " |")') HPS   
   write (unit_o, '(" | Solvation Ext. wall (A)           | DHF    |",7X,f5.2, " |")') DHF   
   write (unit_o, '(" | Added Cutoff (A)                  | DCut   |",7X,f5.2, " |")') DCUT
   write (unit_o, '(" | Electrostatic cutoff (A) (DCUT + DPSExt)   |",7X,f5.2, " |")') sqrt(rcutcoul2)
   write (unit_o, '(" | Solvation cutoff (A)     (DCUT + DHF)      |",7X,f5.2, " |")') sqrt(rcutsolv2)
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | Other                                                    |")')  
   write (unit_o, '(" ------------------------------------------------------------")')
   write (unit_o, '(" | FSolv                                      |",7X,f5.2, " |")') FSOLV
   write (unit_o, '(" | FVdW                                       |",7X,f5.2, " |")') FVDW
   write (unit_o, '(" | Random generator seed                      |",7X,i5  " |")') seed
   write (unit_o, '(" | IDAB, IGOAB, IWR, ISOLV                    |",4X,4i2," |")') IDAB, IGOAB, IWR, ISOLV
   if (IDIMS.eq.1) then
   write (unit_o, '(" | MC Rej. Factor (xbeta)                     |",7X,f5.2, " |")') Xbeta
   write (unit_o, '(" | Convergence limit (sclim)                  |",7X,f5.2, " |")') sclim
   endif
   write (unit_o, '(" ------------------------------------------------------------")')
 end subroutine writeInputParamSet
!===============================================================================
 END MODULE paramSet
