# SIMDYNAMICS APPLICATION

## Folder structure

├── **binary** : contains executable files  
├── **data** : data for sample executions     
├── **discrete_src** : sources for the executables in binary   
├── **pom.xml** : POM file  
├── **README.md** : this MarkDown readme file  
├── **scripts** : launch scripts examples  
└── **src** : SIMDYNAMICS source files  

## Application Description

DISCRETE is a package devised to simulate the dynamics of proteins using the
Discrete Molecular Dynamics (DMD) methods.  In such simulations, the particles
are assumed to move with constant velocity until a collision occurs, conserving the
total momentum and energy, which drastically saves computation time compared to
standard MD protocols.

The  simulation  program  of  DISCRETE  receives  as  input  a  coordinate  and
a  topology  files,  which  are  generated  with  a  setup  program  also  included  in  the
package.  The coordinate file provides the position of each atom in the structure, and
the topology file contains information about the chemical structure of the molecule
and the charge of the atoms.  Besides,  the simulation program reads a parameter
file,  which  basically  specifies  three  values:   EPS  (Coulomb  interactions),  FSOLV
(solvation) and FVDW (Van Der Waals terms).

The SimDynamics application is a sequential Java program that makes use of the DISCRETE package.
Starting from a set of protein structures, the objective of SimDynamics is to find the
values of the EPS, FSOLV and FVDW parameters that minimize the overall energy
obtained when simulating their molecular dynamics with DISCRETE. Hence, SimDynamics is an example of a
parameter-sweeping application:  for each parameter, a fixed number of values within a range is considered
and a set of simulations (one per  structure)  is  performed  for  each  combination  of  these  values
(configuration). Once all the simulations for a specific configuration have completed, the configura-
tion’s score is calculated and later compared to the others in order to find the best
one.

The main program of the SimDynamics application is divided in three phases:
1.  For each of the N input protein structures,  their corresponding topology and
coordinate files are generated.  These files are independent of the values of EPS,
FSOLV and FVDW.
2.  Parameter-sweep  simulations:  a  simulation  is  executed  for  each  configuration
and each structure.  These simulations do not depend on each other.  The more
values evaluated for each parameter, the more accurate will be the solution.
3.  Finding the configuration with minimal energy: the execution of each simulation
outputs a trajectory and an energy file, which are used to calculate a coefficient
for each configuration.  The main result of the application is the configuration
that minimises that coefficient.

## Instructions

Before running the application, you need to compile DMDSetup and Discrete. First, run 'make' in 
**discrete_src/discrete0.2.4** and **discrete_src/setup0.2.1**. Then, copy **discrete_src/exe/DMDSetup0.2.1** 
and **discrete_src/exe/discrete0.2.4** to **binary** as DMDSetup and discrete respectively. 

