"""
GROMACS Tutorial with PyCOMPSs
Lysozyme in Water

This example will guide a new user through the process of setting up a
simulation system containing a set of proteins (lysozymes) in boxes of water,
with ions. Each step will contain an explanation of input and output,
using typical settings for general use.

Extracted from: http://www.mdtutorials.com/gmx/lysozyme/index.html
Originally done by: Justin A. Lemkul, Ph.D.
From: Virginia Tech Department of Biochemistry

This example reaches up to stage 4 (energy minimization).
"""

from os import listdir
from os.path import isfile, join
import sys

from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.binary import binary
from pycompss.api.mpi import mpi
from pycompss.api.parameter import *


# ############ #
# Step 1 tasks #
# ############ #

@binary(binary='${GMX_BIN}/gmx')
@task(protein=FILE_IN,
      structure=FILE_OUT,
      topology=FILE_OUT)
def generate_topology(mode='pdb2gmx',
                         protein_flag='-f', protein=None,
                         structure_flag='-o', structure=None,
                         topology_flag='-p', topology=None,
                         flags='-ignh',
                         forcefield_flag='-ff', forcefield='oplsaa',
                         water_flag='-water', water='spce'):
    # Command: gmx pdb2gmx -f protein.pdb -o structure.gro -p topology.top -ignh -ff amber03 -water tip3p
    pass

# ############ #
# Step 2 tasks #
# ############ #

@binary(binary='${GMX_BIN}/gmx')
@task(structure=FILE_IN,
      structure_newbox=FILE_OUT)
def define_box(mode='editconf',
               structure_flag='-f', structure=None,
               structure_newbox_flag='-o', structure_newbox=None,
               center_flag='-c',
               distance_flag='-d', distance='1.0',
               boxtype_flag='-bt', boxtype='cubic'):
    # Command: gmx editconf -f structure.gro -o structure_newbox.gro -c -d 1.0 -bt cubic
    pass

# ############ #
# Step 3 tasks #
# ############ #

@binary(binary='${GMX_BIN}/gmx')
@task(structure_newbox=FILE_IN,
      protein_solv=FILE_OUT,
      topology=FILE_IN)
def add_solvate(mode='solvate',
                structure_newbox_flag='-cp', structure_newbox=None,
                configuration_solvent_flag='-cs', configuration_solvent='spc216.gro',
                protein_solv_flag='-o', protein_solv=None,
                topology_flag='-p', topology=None):
    # Command: gmx solvate -cp structure_newbox.gro -cs spc216.gro -o protein_solv.gro -p topology.top
    pass

# ############ #
# Step 4 tasks #
# ############ #

@binary(binary='${GMX_BIN}/gmx')
@task(conf=FILE_IN,
      protein_solv=FILE_IN,
      topology=FILE_IN,
      output=FILE_OUT)
def assemble_tpr(mode='grompp',
                conf_flag='-f', conf=None,
                protein_solv_flag='-c', protein_solv=None,
                topology_flag='-p', topology=None,
                output_flag='-o', output=None):
    # Command: gmx grompp -f ions.mdp -c protein_solv.gro -p topology.top -o ions.tpr
    pass

@binary(binary='${GMX_BIN}/gmx')
@task(ions=FILE_IN,
      output=FILE_OUT,
      topology=FILE_IN,
      group={Type:FILE_IN, StdIOStream:STDIN})
def replace_solvent_with_ions(mode='genion',
                              ions_flag='-s', ions=None,
                              output_flag='-o', output=None,
                              topology_flag='-p', topology=None,
                              pname_flag='-pname', pname='NA',
                              nname_flag='-nname', nname='CL',
                              neutral_flag='-neutral',
                              group=None):
    # Command: gmx genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral < ../config/genion.group
    pass

# ############ #
# Step 5 tasks #
# ############ #

computing_units = "24"
computing_nodes = "1"

@constraint(computing_units=computing_units)
@mpi(runner="mpirun", binary="gmx_mpi", computing_nodes=computing_nodes)
@task(em=FILE_IN,
      em_energy=FILE_OUT)
def energy_minimization(mode='mdrun',
                        verbose_flag='-v',
                        ompthreads_flag='-ntomp', ompthreads='0',
                        em_flag='-s', em=None,
                        em_energy_flag='-e', em_energy=None):
    # Command: gmx mdrun -v -s em.tpr
    pass

# ############ #
# Step 6 tasks #
# ############ #

@binary(binary='${GMX_BIN}/gmx')
@task(em=FILE_IN,
      output=FILE_OUT,
      selection={Type:FILE_IN, StdIOStream:STDIN})
def energy_analisis(mode='energy',
                    em_flag='-f', em=None,
                    output_flag='-o', output=None,
                    selection=None):
    # Command: gmx energy -f em.edr -o output.xvg
    pass


# ############# #
# MAIN FUNCTION #
# ############# #

def main(dataset_path, output_path, config_path):
    print("Starting demo")

    protein_names = []
    protein_pdbs = []

    # Look for proteins in the dataset folder
    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            protein_names.append(f.split('.')[0])
            protein_pdbs.append(join(dataset_path, f))
    proteins = zip(protein_names, protein_pdbs)

    # Iterate over the proteins and process them
    result_image_paths = []
    for name, pdb in proteins:
        # 1st step - Generate topology
        structure = join(output_path, name + '.gro')
        topology = join(output_path, name + '.top')
        generate_topology(protein=pdb,
                          structure=structure,
                          topology=topology)
        # 2nd step - Define box
        structure_newbox = join(output_path, name + '_newbox.gro')
        define_box(structure=structure,
                   structure_newbox=structure_newbox)
        # 3rd step - Add solvate
        protein_solv = join(output_path, name + '_solv.gro')
        add_solvate(structure_newbox=structure_newbox,
                    protein_solv=protein_solv,
                    topology=topology)
        # 4th step - Add ions
        # Assemble with ions.mdp
        ions_conf = join(config_path, 'ions.mdp')
        ions = join(output_path, name + '_ions.tpr')
        assemble_tpr(conf=ions_conf,
                     protein_solv=protein_solv,
                     topology=topology,
                     output=ions)
        protein_solv_ions = join(output_path, name + '_solv_ions.gro')
        group = join(config_path, 'genion.group')   # 13 = SOL
        replace_solvent_with_ions(ions=ions,
                                  output=protein_solv_ions,
                                  topology=topology,
                                  group=group)
        # 5th step - Minimize energy
        # Reasemble with minim.mdp
        minim_conf = join(config_path, 'minim.mdp')
        em = join(output_path, name + '_em.tpr')
        assemble_tpr(conf=minim_conf,
                     protein_solv=protein_solv_ions,
                     topology=topology,
                     output=em)
        em_energy = join(output_path, name + '_em_energy.edr')
        energy_minimization(em=em,
                            em_energy=em_energy)
        # 6th step - Energy analysis (generate xvg image)
        energy_result = join(output_path, name + '_potential.xvg')
        energy_selection = join(config_path, 'energy.selection')  # 10 = potential
        energy_analisis(em=em_energy,
                        output=energy_result,
                        selection=energy_selection)


if __name__=='__main__':
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    main(dataset_path, output_path, config_path)
