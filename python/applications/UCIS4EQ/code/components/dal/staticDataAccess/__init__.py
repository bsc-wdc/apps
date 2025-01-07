"""This is init module."""

################################################################################
# Module imports

from . import staticDataAccess as SDA
from . import webDavRepository as WebDavRepo
from . import SSHRepository as SSHRepo
from . import localRepository as LocalRepo

# Define the supported list of repositories
repositories = SDA.SDAProvider()

# Register each data repository
repositories.registerBuilder('BSC_B2DROP', WebDavRepo.BSCB2DROPRepositoryBuilder())
repositories.registerBuilder('BSC_DT', SSHRepo.BSCDTRepositoryBuilder())
repositories.registerBuilder('ETH_DAINT', SSHRepo.ETHDAINTRepositoryBuilder())
repositories.registerBuilder('LOCAL', LocalRepo.LocalRepositoryBuilder())
