/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
!
! CommLine & I/O
!
   integer, parameter :: NFILES = 12
 
   integer unit_i, unit_o, unit_top, unit_r, unit_ener, unit_traj, unit_rst, unit_pdb, unit_targ, unit_v, unit_rstv, unit_input
 
   type(commLineOption) :: files(NFILES) = (/&
   commLineOption("-i",    "param",          "formatted",   "old",     "Settings"),&
   commLineOption("-top",  "topology",       "unformatted", "old",     "Topology"),&
   commLineOption("-r",    "coordinates",    "unformatted", "old",     "Initial coordinates"),&
   commLineOption("-ener", "energy",         "formatted",   "unknown", "Energies"),&
   commLineOption("-traj", "trajectory.pdb", "formatted",   "unknown", "Trajectory (PDB)"),&
   commLineOption("-o",    "log",            "formatted",   "unknown", "Calculation Log"),&
   commLineOption("-x",    "trajectory.crd", "formatted",   "unknown", "Trajectory (CRD)"),&   
   commLineOption("-rst",  "restart",        "unformatted", "unknown", "Restart coordinates"), &
   commLineOption("-targ", "targpdb",        "formatted",   "unknown", "Target (PDB)"), &
   commLineOption("-rstv", "restartVel",     "unformatted", "unknown", "Restart velocities"), &
   commLineOption("-in",   "input",          "formatted",   "unknown", "Input (PDB)"),&
   commLineOption("-v",    "velocities",     "unformatted", "old",     "Initial velocities") &
   /)

! Coordinates and velocities 
   type(pointDP), allocatable :: r(:), rorig(:), rprev(:), rtarg(:), rtargorig(:)
   type(point), allocatable :: v(:), rsp(:) ! Single precision version of r for binary I/O
   real, allocatable :: distat2(:,:), disttarg(:,:), w(:,:), rcov2(:)
   type(point) calcCM, rcm ! C of M   
   character*4 c1

! Step Potentials
   type (stepPotInt), allocatable :: stepPts(:,:)

! Covalent bonds
   real, allocatable :: rb(:,:),drb(:,:)

! Structure
   integer, parameter :: PROT=1, NUC=2, SMALL=3, COMPLEX=4
   integer molType
   character(len=10), parameter :: topVersion = 'v0.2.1.2'
   character(len=10) tv
   character(len=4), allocatable :: atom(:), res(:), atp(:), atomtg(:), restg(:)
   character(len=1), allocatable :: chain(:)
   integer, allocatable :: in(:),ica(:),ico(:), rnum(:)
   logical, allocatable :: istruct(:,:)                   ! true for SSec based interactions
   integer natom, nres, ncovpairs, nhbs, nhelix, nbeta, onatom, nat
   integer, allocatable :: cov(:,:), hbs(:,:), helix(:,:), beta(:,:)
   integer nmol
   integer, allocatable :: molNum(:), molRes(:), mols(:,:)
   logical, allocatable :: dummy(:), frozen(:)
   integer, allocatable :: nid(:),oid(:)
   integer recnatom, recnmol, recnres
   integer, allocatable :: icatarg(:)
 

! Potentials & energies
   real, allocatable :: evdw(:),rvdw(:),qq(:),gfree(:),vol(:),xlamb(:), xm(:), rhc(:), xsum(:,:)
   real, allocatable :: nxm(:) ! llista de masses per atoms reals
   real potlk, rvdwij, xmassa, esolv, calcEkin, ecoul, ekin, epotfis, epotgo, ekin0
   real :: score, scoreprev, score0,pdistab

! Interaction pairs
   integer bpairs
   integer, allocatable :: blist(:,:)
   type(intpList), allocatable :: nblist(:)

! Collisions
   integer :: ierr
   real*8, allocatable :: tpart(:)
   integer, allocatable :: ipart(:)
   integer mem1, mem2, ibloc, iev, npair1, npair2
! MC
   integer iacc, iaccp, idec, idecp
! Time
   real*8 tacact, taccorr, tacum, tevent, tevent1, temps, tacrect, tactot
   real*8 tinit, tsetup, tfin

!
   integer i,j,k, ioerr
!
   real distance
   logical MCCheck
!
   real, parameter :: A = 1.e-10 ! M to Angs
