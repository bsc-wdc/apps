"""
Contains the filenames of the fortran executables used by the FIXED phase
"""
# No need to be a class
# class FortranWrapper(object):

FC = "ifort"
GFC = "gfortran"
MC_FLAG = "-mcmodel=large"
SHARED_FLAG = "-shared-intel"
CONVERT_PREFIX = "-convert"
CONVERT_VALUE = "big_endian"
TRACEBACK_FLAG = "-traceback"
ASSUME_PREFIX = "-assume"
ASSUME_VALUE = "byterecl"
BIG_O_FLAG = "-O"
OPT_FLAG = "-O3"
FPMODEL_PREFIX = "-fp-model"
FPMODEL_VALUE = "precise"
STACK_FLAG = "-fp-stack-check"
CFLAG = "-c"
OFLAG = "-o"
MODULE_FLAG = "-module"

W3_LIB_DIR = "w3lib-2.0.2/"
BACIO_LIB_DIR = "bacio/"
W3_FLAG = "-lw3_4"
BACIO_FLAG = "-lbacio_4"

#EXTRAE_FLAG = "-L/path/to/extrae/libmpitrace.so"

SUFFIX_F90_SRC = ".f90"
SUFFIX_F_SRC = ".f"
SUFFIX_OBJECT = ".o"
SUFFIX_EXE = ".x"

'''
* FIXED FORTRAN FILES
'''
BOTSOILTYPE = "botsoiltype"
GFDLCO2 = "gfdlco2"
DEEPTEMPERATURE = "deeptemperature"
ENVELOPE = "envelope"
LANDUSE = "landuse"
LANDUSENEW = "landusenew"
SMMOUNT = "smmount"
ROUGHNESS = "roughness"
STDH = "stdh"
STDHTOPO = "stdhtopo"
SNOWALBEDO = "snowalbedo"
TOPO = "topo"
TOPOSEAMASK = "toposeamask"
TOPSOILTYPE = "topsoiltype"
VCGENERATOR = "vcgenerator"

FIXED_FORTRAN_F90_FILES = [BOTSOILTYPE, DEEPTEMPERATURE, ENVELOPE, LANDUSE, LANDUSENEW, SMMOUNT, ROUGHNESS, STDH, STDHTOPO, SNOWALBEDO, TOPO, TOPOSEAMASK, TOPSOILTYPE, VCGENERATOR]

FIXED_FORTRAN_F_FILES = [GFDLCO2]

'''
* VARIABLE FORTRAN FILES
'''
ALBEDO = "albedo"
ALBEDO_RRTM_1DEG = "albedorrtm1deg"
ALLPREP_RRTM = "allprep_rrtm"
CNV_RRTM = "cnv_rrtm"
DEGRIB_SST = "degribsst"
DUST_START = "dust_start"
GFS2MODEL = "gfs2model_rrtm"
INC_RRTM = "inc_rrtm"
MODULE_FLT = "module_flt"  # used by all_prep
VEG_FRAC = "vegfrac"
Z0_VEGUSTAR = "z0vegustar"
READ_PAUL_SOURCE = "read_paul_source"  # compiled in script

VARIABLE_FORTRAN_F90_DEP_FILES = [MODULE_FLT]

VARIABLE_FORTRAN_F90_FILES = [ALBEDO, ALBEDO_RRTM_1DEG, VEG_FRAC, Z0_VEGUSTAR]

VARIABLE_FORTRAN_F_FILES = [CNV_RRTM, DUST_START, INC_RRTM]

VARIABLE_GFORTRAN_F_FILES = [GFS2MODEL]

VARIABLE_FORTRAN_F_FILES_WITH_DEPS = [ALLPREP_RRTM]

VARIABLE_FORTRAN_F_FILES_WITH_W3 = [DEGRIB_SST]

'''
* UMO MODEL FORTRAN FILES
'''
NEMS = "NEMS"

'''
* POST PROCESS FORTRAN FILES
'''
NEW_POSTALL = "new_postall"
