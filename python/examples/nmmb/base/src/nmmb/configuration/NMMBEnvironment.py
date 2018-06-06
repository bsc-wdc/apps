import os
import NMMBConstants

"""
Retrieve information from environment variables
* Separators at the end of the paths are required.
"""

UMO_PATH = os.environ[NMMBConstants.ENV_NAME_UMO_PATH] + os.path.sep
UMO_ROOT = os.environ[NMMBConstants.ENV_NAME_UMO_ROOT] + os.path.sep

FIX = os.environ[NMMBConstants.ENV_NAME_FIX] + os.path.sep
VRB = os.environ[NMMBConstants.ENV_NAME_VRB] + os.path.sep
POST_CARBONO = os.environ[NMMBConstants.ENV_NAME_POST_CARBONO] + os.path.sep

OUTPUT = os.environ[NMMBConstants.ENV_NAME_OUTPUT] + os.path.sep
OUTNMMB = os.environ[NMMBConstants.ENV_NAME_OUTNMMB] + os.path.sep
UMO_OUT = os.environ[NMMBConstants.ENV_NAME_UMO_OUT] + os.path.sep

SRCDIR = os.environ[NMMBConstants.ENV_NAME_SRCDIR] + os.path.sep
CHEMIC = os.environ[NMMBConstants.ENV_NAME_CHEMIC] + os.path.sep
DATMOD = os.environ[NMMBConstants.ENV_NAME_DATMOD] + os.path.sep

FNL = os.environ[NMMBConstants.ENV_NAME_FNL] + os.path.sep
GFS = os.environ[NMMBConstants.ENV_NAME_GFS] + os.path.sep

# Infer some FIXED paths
GEODATA_DIR = FIX + ".." + os.path.sep + "geodata" + os.path.sep
GTOPO_DIR = FIX + ".." + os.path.sep + "GTOPO30" + os.path.sep
FIX_INCLUDE_DIR = FIX + "include" + os.path.sep

# Infer some VARIABLE paths
VRB_INCLUDE_DIR = VRB + "include" + os.path.sep

# Infer some UMO Model paths
UMO_LIBS = SRCDIR + "libs" + os.path.sep
