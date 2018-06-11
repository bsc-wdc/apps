from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import *
from pycompss.api.mpi import mpi

import configuration.NMMBConstants as NMMBConstants
import utils.FortranWrapper as FortranWrapper


@constraint(computingUnits="${NEMS_CUS_PER_NODE}")  # "16")
@mpi(runner="mpirun",
     binary="${" + NMMBConstants.ENV_NAME_SRCDIR + "}/exe/" + FortranWrapper.NEMS + FortranWrapper.SUFFIX_EXE,
     workingDir="${" + NMMBConstants.ENV_NAME_UMO_OUT + "}/",
     computingNodes="${NEMS_NODES}")  # 4)
@task(returns=int, stdOutFile=FILE_OUT_STDOUT, stdErrFile=FILE_OUT_STDERR)
def nems(stdOutFile, stdErrFile):
    pass
