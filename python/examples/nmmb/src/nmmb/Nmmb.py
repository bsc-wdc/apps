import os
import sys
import time
import logging
import datetime
import loggers.LoggerNames as LoggerNames
import configuration.NMMBEnvironment as NMMBEnvironment
from configuration.NMMBConfigManager import NMMBConfigManager
from configuration.NMMBParameters import NMMBParameters
import configuration.NMMBConstants as NMMBConstants
from userexceptions.MainExecutionException import MainExecutionException
from userexceptions.TaskExecutionException import TaskExecutionException
from utils.MessagePrinter import MessagePrinter
import utils.FileManagement as FileManagement
import utils.FortranWrapper as FortranWrapper
import binary.BINARY as BINARY
import mpi.MPI as MPI
from plot.plotter import generate_figures
from plot.giffer import generate_animation

from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_open

# Loggers
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
LOGGER_MAIN = logging.getLogger(LoggerNames.NMMB_MAIN)
LOGGER_FIXED = logging.getLogger(LoggerNames.NMMB_FIXED)
LOGGER_VARIABLE = logging.getLogger(LoggerNames.NMMB_VARIABLE)
LOGGER_UMO_MODEL = logging.getLogger(LoggerNames.NMMB_UMO_MODEL)
LOGGER_POST = logging.getLogger(LoggerNames.NMMB_POST)
LOGGER_IMAGE = logging.getLogger(LoggerNames.NMMB_IMAGE)
LOGGER_ANIMATION = logging.getLogger(LoggerNames.NMMB_ANIMATION)


def usage():
    """
    Print usage
    """
    LOGGER_MAIN.info("Invalid parameters for nmmb.Nmmb")
    LOGGER_MAIN.info("    Usage: nmmb.Nmmb <configFilePath>")


'''
* ***************************************************************************************************
* ***************************************************************************************************
* ***************************************************************************************************
* ******************** FIXED STEP *******************************************************************
* ***************************************************************************************************
* ***************************************************************************************************
* ***************************************************************************************************
'''


def doFixed(nmmbParams):
    """
    Do fixed step processing.
    :param nmmbParams: Nmmb parameters
    :raise TaskExecutionException: When a task fails.
    """
    LOGGER_FIXED.info("Enter fixed process")

    # Prepare execution *************************************************************
    nmmbParams.prepareFixedExecution()
    fixedMP = MessagePrinter(LOGGER_FIXED)

    # Build the fortran executables ************************************************
    if nmmbParams.isCompileBinaries():
        compilationEvs = []
        fixedMP.printInfoMsg("Building fixed executables")
        for fortranFile in FortranWrapper.FIXED_FORTRAN_F90_FILES:
            executable = NMMBEnvironment.FIX + fortranFile + FortranWrapper.SUFFIX_EXE
            src = NMMBEnvironment.FIX + fortranFile + FortranWrapper.SUFFIX_F90_SRC
            compilationEvs.append(BINARY.fortranCompiler(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                         FortranWrapper.CONVERT_PREFIX, FortranWrapper.CONVERT_VALUE,
                                                         FortranWrapper.TRACEBACK_FLAG, FortranWrapper.ASSUME_PREFIX,
                                                         FortranWrapper.ASSUME_VALUE, FortranWrapper.OPT_FLAG,
                                                         FortranWrapper.FPMODEL_PREFIX, FortranWrapper.FPMODEL_VALUE,
                                                         FortranWrapper.STACK_FLAG, FortranWrapper.OFLAG,
                                                         executable, src))

        for fortranFile in FortranWrapper.FIXED_FORTRAN_F_FILES:
            executable = NMMBEnvironment.FIX + fortranFile + FortranWrapper.SUFFIX_EXE
            src = NMMBEnvironment.FIX + fortranFile + FortranWrapper.SUFFIX_F_SRC
            compilationEvs.append(BINARY.fortranCompiler(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                         FortranWrapper.CONVERT_PREFIX, FortranWrapper.CONVERT_VALUE,
                                                         FortranWrapper.TRACEBACK_FLAG, FortranWrapper.ASSUME_PREFIX,
                                                         FortranWrapper.ASSUME_VALUE, FortranWrapper.OPT_FLAG,
                                                         FortranWrapper.FPMODEL_PREFIX, FortranWrapper.FPMODEL_VALUE,
                                                         FortranWrapper.STACK_FLAG, FortranWrapper.OFLAG,
                                                         executable, src))
        # Sync master to wait for compilation
        compilationEvs = compss_wait_on(compilationEvs)
        i = 0
        for cEv in compilationEvs:
            LOGGER_FIXED.debug("Compilation of " + str(i) + " binary ended with status " + str(cEv))
            if cEv != 0:
                raise TaskExecutionException("[ERROR] Error compiling binary " + str(i))
            i += 1
        fixedMP.printInfoMsg("Finished building fixed executables")

    # Begin binary calls **********************************************************
    fixedMP.printHeaderMsg("BEGIN")

    NUM_BINARIES = 16
    fixedBinariesEvs = []

    fixedMP.printInfoMsg("Generate DEM height and sea mask files")
    topoDir = NMMBEnvironment.GEODATA_DIR + "topo1kmDEM" + os.path.sep
    seamaskDEM = NMMBEnvironment.OUTPUT + "seamaskDEM"
    heightDEM = NMMBEnvironment.OUTPUT + "heightDEM"
    fixedBinariesEvs.append(BINARY.smmount(topoDir, seamaskDEM, heightDEM))

    fixedMP.printInfoMsg("Generate landuse file")
    landuseDataDir = NMMBEnvironment.GEODATA_DIR + "landuse_30s" + os.path.sep
    landuse = NMMBEnvironment.OUTPUT + "landuse"
    kount_landuse = NMMBEnvironment.OUTPUT + "kount_landuse"
    fixedBinariesEvs.append(BINARY.landuse(landuseDataDir, landuse, kount_landuse))

    fixedMP.printInfoMsg("Generate landusenew file")
    landusenew = NMMBEnvironment.OUTPUT + "landusenew"
    kount_landusenew = NMMBEnvironment.OUTPUT + "kount_landusenew"
    fixedBinariesEvs.append(BINARY.landusenew(NMMBEnvironment.GTOPO_DIR, landusenew, kount_landusenew))

    fixedMP.printInfoMsg("Generate mountains")
    topo30sDir = NMMBEnvironment.GEODATA_DIR + "topo_30s" + os.path.sep
    heightmean = NMMBEnvironment.OUTPUT + "heightmean"
    fixedBinariesEvs.append(BINARY.topo(topo30sDir, heightmean))

    fixedMP.printInfoMsg("Generate standard deviation of topography height")
    stdh = NMMBEnvironment.OUTPUT + "stdh"
    fixedBinariesEvs.append(BINARY.stdh(heightmean, seamaskDEM, topo30sDir, stdh))

    fixedMP.printInfoMsg("Generate envelope mountains")
    height = NMMBEnvironment.OUTPUT + "height"
    fixedBinariesEvs.append(BINARY.envelope(heightmean, stdh, height))

    fixedMP.printInfoMsg("Generate top soil type file")
    soiltypeDir = NMMBEnvironment.GEODATA_DIR + "soiltype_top_30s" + os.path.sep
    topsoiltype = NMMBEnvironment.OUTPUT + "topsoiltype"
    fixedBinariesEvs.append(BINARY.topsoiltype(seamaskDEM, soiltypeDir, topsoiltype))

    fixedMP.printInfoMsg("Generate bottom soil type file")
    soiltypePath = NMMBEnvironment.GEODATA_DIR + "soiltype_bot_30s" + os.path.sep
    botsoiltype = NMMBEnvironment.OUTPUT + "botsoiltype"
    fixedBinariesEvs.append(BINARY.botsoiltype(seamaskDEM, soiltypePath, botsoiltype))

    fixedMP.printInfoMsg("Generate sea mask and reprocess mountains")
    seamask = NMMBEnvironment.OUTPUT + "seamask"
    fixedBinariesEvs.append(BINARY.toposeamask(seamaskDEM, seamask, height, landuse, topsoiltype, botsoiltype))

    fixedMP.printInfoMsg("Reprocess standard deviation of topography height")
    fixedBinariesEvs.append(BINARY.stdhtopo(seamask, stdh))

    fixedMP.printInfoMsg("Generate deep soil temperature")
    soiltempPath = NMMBEnvironment.GEODATA_DIR + "soiltemp_1deg" + os.path.sep
    deeptemperature = NMMBEnvironment.OUTPUT + "deeptemperature"
    fixedBinariesEvs.append(BINARY.deeptemperature(seamask, soiltempPath, deeptemperature))

    fixedMP.printInfoMsg("Generate maximum snow albedo")
    maxsnowalbDir = NMMBEnvironment.GEODATA_DIR + "maxsnowalb" + os.path.sep
    snowalbedo = NMMBEnvironment.OUTPUT + "snowalbedo"
    fixedBinariesEvs.append(BINARY.snowalbedo(maxsnowalbDir, snowalbedo))

    fixedMP.printInfoMsg("Generate vertical coordinate")
    dsg = NMMBEnvironment.OUTPUT + "dsg"
    fixedBinariesEvs.append(BINARY.vcgenerator(dsg))

    fixedMP.printInfoMsg("Generate highres roughness length for africa and asia")
    roughnessDir = NMMBEnvironment.GEODATA_DIR + "roughness_025s" + os.path.sep
    roughness = NMMBEnvironment.OUTPUT + "roughness"
    fixedBinariesEvs.append(BINARY.roughness(roughnessDir, roughness))

    fixedMP.printInfoMsg("Generate co2 files")
    co2_data_dir = NMMBEnvironment.GEODATA_DIR + "co2data" + os.path.sep
    co2_trans = NMMBEnvironment.OUTPUT + "co2_trans"
    fixedBinariesEvs.append(BINARY.gfdlco2(dsg, co2_data_dir, co2_trans))

    fixedMP.printInfoMsg("Generate lookup tables for aerosol scavenging collection efficiencies")
    lookup_aerosol2_rh00 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh00"
    lookup_aerosol2_rh50 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh50"
    lookup_aerosol2_rh70 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh70"
    lookup_aerosol2_rh80 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh80"
    lookup_aerosol2_rh90 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh90"
    lookup_aerosol2_rh95 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh95"
    lookup_aerosol2_rh99 = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh99"
    fixedBinariesEvs.append(BINARY.run_aerosol(nmmbParams.isCompileBinaries(), nmmbParams.isCleanBinaries(),
                                               lookup_aerosol2_rh00, lookup_aerosol2_rh50, lookup_aerosol2_rh70,
                                               lookup_aerosol2_rh80, lookup_aerosol2_rh90, lookup_aerosol2_rh95,
                                               lookup_aerosol2_rh99))

    # Wait for binaries completion and check exit value ****************************
    fixedBinariesEvs = compss_wait_on(fixedBinariesEvs)
    i = 0
    for fBEV in fixedBinariesEvs:
        LOGGER_FIXED.debug("Execution of " + str(i) + " binary ended with status " + str(fBEV))
        if fBEV != 0:
            raise TaskExecutionException("[ERROR] Error executing binary " + i)
        i += 1

    # Previously done within the copyFilesFromPreprocess function.
    # Clean specific files
    outputFiles = ["ETAMPNEW_AERO"]
    for of in outputFiles:
        filePath = NMMBEnvironment.UMO_OUT + of
        if not FileManagement.deleteFile(filePath):
            LOGGER_UMO_MODEL.debug("Cannot erase previous " + of + " because it doesn't exist.")

    LOGGER_FIXED.info("Copy the files from this process.")
    lookupAerosol2RH00Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH00"
    with open(lookupAerosol2RH00Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh00)
        dest.write(orig.read())  # We will need this file later
        orig.close()
    lookupAerosol2RH50Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH50"
    with open(lookupAerosol2RH50Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh50)
        dest.write(orig.read())  # We will need this file later
        orig.close()
    lookupAerosol2RH70Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH70"
    with open(lookupAerosol2RH70Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh70)
        dest.write(orig.read())  # We will need this file later
        orig.close()
    lookupAerosol2RH80Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH80"
    with open(lookupAerosol2RH80Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh80)
        dest.write(orig.read())  # We will need this file later
        orig.close()
    lookupAerosol2RH90Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH90"
    with open(lookupAerosol2RH90Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh90)
        dest.write(orig.read())  # We will need this file later
        orig.close()
    lookupAerosol2RH95Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH95"
    with open(lookupAerosol2RH95Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh95)
        dest.write(orig.read())  # We will need this file later
        orig.close()
    lookupAerosol2RH99Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH99"
    with open(lookupAerosol2RH99Target, 'w') as dest:
        orig = compss_open(lookup_aerosol2_rh99)
        dest.write(orig.read())  # We will need this file later
        orig.close()

    # Clean Up binaries ***********************************************************
    if nmmbParams.isCleanBinaries():
        fixedMP.printInfoMsg("Clean up executables")
        for fortranFile in FortranWrapper.FIXED_FORTRAN_F90_FILES:
            executable = NMMBEnvironment.FIX + fortranFile + FortranWrapper.SUFFIX_EXE
            if os.path.exists(executable):
                os.remove(executable)
        for fortranFile in FortranWrapper.FIXED_FORTRAN_F_FILES:
            executable = NMMBEnvironment.FIX + fortranFile + FortranWrapper.SUFFIX_EXE
            if os.path.exists(executable):
                os.remove(executable)
    # End ************************************************************************
    fixedMP.printHeaderMsg("END")

    LOGGER_FIXED.info("Fixed process finished")


'''
 * ***************************************************************************************************
 * ***************************************************************************************************
 * ***************************************************************************************************
 * ******************** VARIABLE STEP ****************************************************************
 * ***************************************************************************************************
 * ***************************************************************************************************
 * ***************************************************************************************************
'''


def compileVariable():
    """
    Compile binaries for the variable step.
    :raise TaskExecutionException: When a task fails.
    """
    # Build the fortran objects *************************************************
    depCompilationEvs = []
    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F90_DEP_FILES:
        moduleDir = NMMBEnvironment.VRB
        object = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_OBJECT
        src = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_F90_SRC
        depCompilationEvs.append(BINARY.fortranCompileObject(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                             FortranWrapper.CONVERT_PREFIX,
                                                             FortranWrapper.CONVERT_VALUE,
                                                             FortranWrapper.TRACEBACK_FLAG,
                                                             FortranWrapper.ASSUME_PREFIX, FortranWrapper.ASSUME_VALUE,
                                                             FortranWrapper.OPT_FLAG, FortranWrapper.FPMODEL_PREFIX,
                                                             FortranWrapper.FPMODEL_VALUE, FortranWrapper.STACK_FLAG,
                                                             FortranWrapper.CFLAG, src, FortranWrapper.OFLAG,
                                                             object, FortranWrapper.MODULE_FLAG, moduleDir))

    # Sync to check compilation status (dependency with task object is also respected if this sync is erased)
    depCompilationEvs = compss_wait_on(depCompilationEvs)
    i = 0
    for cEv in depCompilationEvs:
        LOGGER_VARIABLE.debug("Compilation of " + str(i) + " dependant binary ended with status " + str(cEv))
        if cEv != 0:
            raise TaskExecutionException("[ERROR] Error compiling binary " + str(i))
        i += 1

    # Build the fortran executables *************************************************
    compilationEvs = []

    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F90_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        src = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_F90_SRC
        compilationEvs.append(BINARY.fortranCompiler(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                     FortranWrapper.CONVERT_PREFIX, FortranWrapper.CONVERT_VALUE,
                                                     FortranWrapper.TRACEBACK_FLAG, FortranWrapper.ASSUME_PREFIX,
                                                     FortranWrapper.ASSUME_VALUE, FortranWrapper.OPT_FLAG,
                                                     FortranWrapper.FPMODEL_PREFIX, FortranWrapper.FPMODEL_VALUE,
                                                     FortranWrapper.STACK_FLAG, FortranWrapper.OFLAG,
                                                     executable, src))

    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        src = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_F_SRC
        compilationEvs.append(BINARY.fortranCompiler(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                     FortranWrapper.CONVERT_PREFIX, FortranWrapper.CONVERT_VALUE,
                                                     FortranWrapper.TRACEBACK_FLAG, FortranWrapper.ASSUME_PREFIX,
                                                     FortranWrapper.ASSUME_VALUE, FortranWrapper.OPT_FLAG,
                                                     FortranWrapper.FPMODEL_PREFIX, FortranWrapper.FPMODEL_VALUE,
                                                     FortranWrapper.STACK_FLAG, FortranWrapper.OFLAG,
                                                     executable, src))

    for fortranFile in FortranWrapper.VARIABLE_GFORTRAN_F_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        src = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_F_SRC
        compilationEvs.append(BINARY.gfortranCompiler(FortranWrapper.BIG_O_FLAG, src, FortranWrapper.OFLAG,
                                                      executable))

    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F_FILES_WITH_W3:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        src = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_F_SRC
        w3LibFlag = "-L" + NMMBEnvironment.UMO_LIBS + FortranWrapper.W3_LIB_DIR
        bacioLibFlag = "-L" + NMMBEnvironment.UMO_LIBS + FortranWrapper.BACIO_LIB_DIR
        compilationEvs.append(BINARY.fortranCompilerWithW3(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                           FortranWrapper.CONVERT_PREFIX, FortranWrapper.CONVERT_VALUE,
                                                           FortranWrapper.TRACEBACK_FLAG, FortranWrapper.ASSUME_PREFIX,
                                                           FortranWrapper.ASSUME_VALUE, FortranWrapper.OPT_FLAG,
                                                           FortranWrapper.FPMODEL_PREFIX, FortranWrapper.FPMODEL_VALUE,
                                                           FortranWrapper.STACK_FLAG, FortranWrapper.OFLAG,
                                                           executable, src, w3LibFlag, bacioLibFlag,
                                                           FortranWrapper.W3_FLAG, FortranWrapper.BACIO_FLAG))

    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F_FILES_WITH_DEPS:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        src = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_F_SRC
        object = NMMBEnvironment.VRB + FortranWrapper.MODULE_FLT + FortranWrapper.SUFFIX_OBJECT
        compilationEvs.append(BINARY.fortranCompileWithObject(FortranWrapper.MC_FLAG, FortranWrapper.SHARED_FLAG,
                                                              FortranWrapper.CONVERT_PREFIX,
                                                              FortranWrapper.CONVERT_VALUE,
                                                              FortranWrapper.TRACEBACK_FLAG,
                                                              FortranWrapper.ASSUME_PREFIX,
                                                              FortranWrapper.ASSUME_VALUE, FortranWrapper.OPT_FLAG,
                                                              FortranWrapper.FPMODEL_PREFIX,
                                                              FortranWrapper.FPMODEL_VALUE, FortranWrapper.STACK_FLAG,
                                                              FortranWrapper.OFLAG, executable, src, object))

    source = NMMBEnvironment.VRB + FortranWrapper.READ_PAUL_SOURCE + FortranWrapper.SUFFIX_F90_SRC
    executable = NMMBEnvironment.VRB + FortranWrapper.READ_PAUL_SOURCE + FortranWrapper.SUFFIX_EXE
    compilationEvs.append(BINARY.compileReadPaulSource(source, executable))

    # Sync master to wait for compilation
    compilationEvs = compss_wait_on(compilationEvs)
    i = 0
    for cEv in compilationEvs:
        LOGGER_VARIABLE.debug("Compilation of " + str(i) + " binary ended with status " + str(cEv))
        if cEv != 0:
            raise TaskExecutionException("[ERROR] Error compiling binary " + str(i))
        i += 1


def cleanUpVariableExe():
    """
    Cleanup unused files.
    """
    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F90_DEP_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)
    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F90_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)
    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)
    for fortranFile in FortranWrapper.VARIABLE_GFORTRAN_F_FILES:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)
    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F_FILES_WITH_W3:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)
    for fortranFile in FortranWrapper.VARIABLE_FORTRAN_F_FILES_WITH_DEPS:
        executable = NMMBEnvironment.VRB + fortranFile + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)
    readPaulSource = NMMBEnvironment.VRB + FortranWrapper.READ_PAUL_SOURCE + FortranWrapper.SUFFIX_EXE
    if os.path.exists(readPaulSource):
        os.remove(readPaulSource)


def doVariable(nmmbParams, currentDate):
    """
    Do variable step processing.
    :param nmmbParams: Nmmb parameters
    :param currentDate: Current date (datetime)
    :raise TaskExecutionException: When a task fails.
    """
    LOGGER_VARIABLE.info("Enter variable process")

    # Prepare execution **************************************************************
    nmmbParams.prepareVariableExecution(currentDate)
    variableMP = MessagePrinter(LOGGER_VARIABLE)

    # Compile ************************************************************************
    if nmmbParams.isCompileBinaries():
        variableMP.printInfoMsg("Building variable executables")
        compileVariable()
        variableMP.printInfoMsg("Finished building variable executables")

    # Set variables for binary calls *************************************************
    variableMP.printHeaderMsg("BEGIN")

    CW = NMMBEnvironment.OUTPUT + "00_CW.dump"
    ICEC = NMMBEnvironment.OUTPUT + "00_ICEC.dump"
    SH = NMMBEnvironment.OUTPUT + "00_SH.dump"
    SOILT2 = NMMBEnvironment.OUTPUT + "00_SOILT2.dump"
    SOILT4 = NMMBEnvironment.OUTPUT + "00_SOILT4.dump"
    SOILW2 = NMMBEnvironment.OUTPUT + "00_SOILW2.dump"
    SOILW4 = NMMBEnvironment.OUTPUT + "00_SOILW4.dump"
    TT = NMMBEnvironment.OUTPUT + "00_TT.dump"
    VV = NMMBEnvironment.OUTPUT + "00_VV.dump"
    HH = NMMBEnvironment.OUTPUT + "00_HH.dump"
    PRMSL = NMMBEnvironment.OUTPUT + "00_PRMSL.dump"
    SOILT1 = NMMBEnvironment.OUTPUT + "00_SOILT1.dump"
    SOILT3 = NMMBEnvironment.OUTPUT + "00_SOILT3.dump"
    SOILW1 = NMMBEnvironment.OUTPUT + "00_SOILW1.dump"
    SOILW3 = NMMBEnvironment.OUTPUT + "00_SOILW3.dump"
    SST_TS = NMMBEnvironment.OUTPUT + "00_SST_TS.dump"
    UU = NMMBEnvironment.OUTPUT + "00_UU.dump"
    WEASD = NMMBEnvironment.OUTPUT + "00_WEASD.dump"

    GFS_file = NMMBEnvironment.OUTPUT + "131140000.gfs"

    deco = NMMBEnvironment.VRB

    llspl000 = NMMBEnvironment.OUTPUT + "llspl.000"
    outtmp = NMMBEnvironment.OUTPUT + "llstmp"
    outmst = NMMBEnvironment.OUTPUT + "llsmst"
    outsst = NMMBEnvironment.OUTPUT + "llgsst"
    outsno = NMMBEnvironment.OUTPUT + "llgsno"
    outcic = NMMBEnvironment.OUTPUT + "llgcic"

    llgsst05 = NMMBEnvironment.OUTPUT + "llgsst05"
    sstfileinPath = NMMBEnvironment.OUTPUT + "sst2dvar_grb_0.5"

    seamask = NMMBEnvironment.OUTPUT + "seamask"
    albedo = NMMBEnvironment.OUTPUT + "albedo"
    albedobase = NMMBEnvironment.OUTPUT + "albedobase"
    albedomnth = NMMBEnvironment.GEODATA_DIR + "albedo" + os.path.sep + "albedomnth"

    albedorrtm = NMMBEnvironment.OUTPUT + "albedorrtm"
    albedorrtm1degDir = NMMBEnvironment.GEODATA_DIR + "albedo_rrtm1deg" + os.path.sep

    vegfrac = NMMBEnvironment.OUTPUT + "vegfrac"
    vegfracmnth = NMMBEnvironment.GEODATA_DIR + "vegfrac" + os.path.sep + "vegfracmnth"

    landuse = NMMBEnvironment.OUTPUT + "landuse"
    topsoiltype = NMMBEnvironment.OUTPUT + "topsoiltype"
    height = NMMBEnvironment.OUTPUT + "height"
    stdh = NMMBEnvironment.OUTPUT + "stdh"
    z0base = NMMBEnvironment.OUTPUT + "z0base"
    z0 = NMMBEnvironment.OUTPUT + "z0"
    ustar = NMMBEnvironment.OUTPUT + "ustar"

    sst05 = NMMBEnvironment.OUTPUT + "sst05"
    deeptemperature = NMMBEnvironment.OUTPUT + "deeptemperature"
    snowalbedo = NMMBEnvironment.OUTPUT + "snowalbedo"
    landusenew = NMMBEnvironment.OUTPUT + "landusenew"
    llgsst = NMMBEnvironment.OUTPUT + "llgsst"
    llgsno = NMMBEnvironment.OUTPUT + "llgsno"
    llgcic = NMMBEnvironment.OUTPUT + "llgcic"
    llsmst = NMMBEnvironment.OUTPUT + "llsmst"
    llstmp = NMMBEnvironment.OUTPUT + "llstmp"
    albedorrtmcorr = NMMBEnvironment.OUTPUT + "albedorrtmcorr"
    dzsoil = NMMBEnvironment.OUTPUT + "dzsoil"
    tskin = NMMBEnvironment.OUTPUT + "tskin"
    sst = NMMBEnvironment.OUTPUT + "sst"
    snow = NMMBEnvironment.OUTPUT + "snow"
    snowheight = NMMBEnvironment.OUTPUT + "snowheight"
    cice = NMMBEnvironment.OUTPUT + "cice"
    seamaskcorr = NMMBEnvironment.OUTPUT + "seamaskcorr"
    landusecorr = NMMBEnvironment.OUTPUT + "landusecorr"
    landusenewcorr = NMMBEnvironment.OUTPUT + "landusenewcorr"
    topsoiltypecorr = NMMBEnvironment.OUTPUT + "topsoiltypecorr"
    vegfraccorr = NMMBEnvironment.OUTPUT + "vegfraccorr"
    z0corr = NMMBEnvironment.OUTPUT + "z0corr"
    z0basecorr = NMMBEnvironment.OUTPUT + "z0basecorr"
    emissivity = NMMBEnvironment.OUTPUT + "emissivity"
    canopywater = NMMBEnvironment.OUTPUT + "canopywater"
    frozenprecratio = NMMBEnvironment.OUTPUT + "frozenprecratio"
    smst = NMMBEnvironment.OUTPUT + "smst"
    sh2o = NMMBEnvironment.OUTPUT + "sh2o"
    stmp = NMMBEnvironment.OUTPUT + "stmp"
    dsg = NMMBEnvironment.OUTPUT + "dsg"
    fcst = NMMBEnvironment.OUTPUT + "fcst"
    fcstDir = NMMBEnvironment.OUTPUT + "fcst"
    bocoPrefix = NMMBEnvironment.OUTPUT + "boco."
    llsplPrefix = NMMBEnvironment.OUTPUT + "llspl."

    source = NMMBEnvironment.OUTPUT + "source"
    sourceNETCDF = NMMBEnvironment.OUTPUT + "source.nc"
    sourceNCIncludeDir = NMMBEnvironment.VRB_INCLUDE_DIR

    soildust = NMMBEnvironment.OUTPUT + "soildust"
    kount_landuse = NMMBEnvironment.OUTPUT + "kount_landuse"
    kount_landusenew = NMMBEnvironment.OUTPUT + "kount_landusenew"
    roughness = NMMBEnvironment.OUTPUT + "roughness"

    # Begin binary calls ***********************************************************
    variableBinariesEvs = []

    variableMP.printInfoMsg("degrib gfs global data")
    variableBinariesEvs.append(BINARY.degribgfs_generic_05(CW, ICEC, SH, SOILT2, SOILT4, SOILW2, SOILW4, TT, VV, HH,
                                                           PRMSL, SOILT1, SOILT3, SOILW1, SOILW3, SST_TS, UU, WEASD))

    variableMP.printInfoMsg("GFS 2 Model")
    variableBinariesEvs.append(BINARY.gfs2model_rrtm(CW, ICEC, SH, SOILT2, SOILT4, SOILW2, SOILW4, TT, VV, HH, PRMSL,
                                                     SOILT1, SOILT3, SOILW1, SOILW3, SST_TS, UU, WEASD, GFS_file))

    variableMP.printInfoMsg("INC RRTM")
    variableBinariesEvs.append(BINARY.inc_rrtm(GFS_file, deco))

    variableMP.printInfoMsg("CNV RRTM")
    variableBinariesEvs.append(BINARY.cnv_rrtm(GFS_file, llspl000, outtmp, outmst, outsst, outsno, outcic))

    variableMP.printInfoMsg("Degrib 0.5 deg sst")
    variableBinariesEvs.append(BINARY.degribsst(llgsst05, sstfileinPath))

    variableMP.printInfoMsg("Prepare climatological albedo")
    variableBinariesEvs.append(BINARY.albedo(llspl000, seamask, albedo, albedobase, albedomnth))

    variableMP.printInfoMsg("Prepare rrtm climatological albedos")
    variableBinariesEvs.append(BINARY.albedorrtm(llspl000, seamask, albedorrtm, albedorrtm1degDir))

    variableMP.printInfoMsg("Prepare climatological vegetation fraction")
    variableBinariesEvs.append(BINARY.vegfrac(llspl000, seamask, vegfrac, vegfracmnth))

    variableMP.printInfoMsg("Prepare z0 and initial ustar")
    variableBinariesEvs.append(BINARY.z0vegfrac(seamask, landuse, topsoiltype, height, stdh, vegfrac, z0base, z0,
                                                ustar))

    variableMP.printInfoMsg("Interpolate to model grid and execute allprep (fcst)")
    variableBinariesEvs.append(BINARY.allprep(llspl000, llgsst05, sst05, height, seamask, stdh, deeptemperature,
                                              snowalbedo, z0, z0base, landuse, landusenew, topsoiltype, vegfrac,
                                              albedorrtm, llgsst, llgsno, llgcic, llsmst, llstmp, albedorrtmcorr,
                                              dzsoil, tskin, sst, snow, snowheight, cice, seamaskcorr, landusecorr,
                                              landusenewcorr, topsoiltypecorr, vegfraccorr, z0corr, z0basecorr,
                                              emissivity, canopywater, frozenprecratio, smst, sh2o, stmp, dsg, fcst,
                                              albedo, ustar, fcstDir, bocoPrefix, llsplPrefix))

    variableMP.printInfoMsg("Prepare the dust related variable (soildust)")
    variableBinariesEvs.append(BINARY.readpaulsource(seamask, source, sourceNETCDF, sourceNCIncludeDir))

    variableMP.printInfoMsg("Dust Start")
    variableBinariesEvs.append(BINARY.dust_start(llspl000, soildust, snow, topsoiltypecorr, landusecorr,
                                                 landusenewcorr, kount_landuse, kount_landusenew, vegfrac, height,
                                                 seamask, source, z0corr, roughness))

    # Wait for binaries completion and check exit value *****************************
    variableBinariesEvs = compss_wait_on(variableBinariesEvs)
    i = 0
    for vbEv in variableBinariesEvs:
        LOGGER_VARIABLE.debug("Execution of " + str(i) + " binary ended with status " + str(vbEv))
        if vbEv != 0:
            raise TaskExecutionException("[ERROR] Error executing binary " + str(i))
        i += 1

    variableMP.printHeaderMsg("END")

    # Clean Up binaries ************************************************************
    if nmmbParams.isCleanBinaries():
        variableMP.printInfoMsg("Clean up executables")
        cleanUpVariableExe()

    # Post execution **************************************************************
    folderOutputCase = NMMBEnvironment.OUTNMMB + nmmbParams.getCase() + os.path.sep
    nmmbParams.postVariableExecution(folderOutputCase)

    LOGGER_VARIABLE.info("Variable process finished")

    # Return files to be copied later
    return fcst, soildust


def copyFilesFromPreprocess(nmmbParams, currentDate, fcst, soildust):
    """
    Copy necessary files from preprocess
    :param nmmbParams: Nmmb parameters
    :param currentDate: Current date
    :param fcst: Fcst
    :param soildust: Soil dust
    :raise TaskExecutionException: When a copy fail.
    """
    # Clean specific files
    outputFiles = ["isop.dat", "meteo-data.dat", "chemic-reg", "main_input_filename", "main_input_filename2",
                   "GWD.bin", "configure_file", "co2_trans"]  # , "ETAMPNEW_AERO", "ETAMPNEW_DATA"]
    for of in outputFiles:
        filePath = NMMBEnvironment.UMO_OUT + of
        if not FileManagement.deleteFile(filePath):
            LOGGER_UMO_MODEL.debug("Cannot erase previous " + of + " because it doesn't exist.")

    # Clean regular expr files
    for f in os.listdir(NMMBEnvironment.UMO_OUT):
        if ((f.startswith("lai") and f.endswith(".dat"))
                or (f.startswith("pftp_") and f.endswith(".dat"))
                or (f.startswith("PET") and f.endswith("txt"))
                or (f.startswith("PET") and f.endswith("File")) or (f.startswith("boco."))
                or (f.startswith("boco_chem.")) and (f.startswith("nmm_b_history."))
                or (f.startswith("tr")) or (f.startswith("RRT")) and (f.endswith(".TBL"))
                or (f.startswith("fcstdone.")) or (f.startswith("restartdone."))
                or (f.startswith("nmmb_rst_")) or (f.startswith("nmmb_hst_"))):
            if not FileManagement.deleteFile(f):
                LOGGER_UMO_MODEL.debug("Cannot erase previous " + f + " because it doesn't exist.")

    # Prepare UMO model files
    outputFolderPath = NMMBEnvironment.OUTPUT
    for f in os.listdir(outputFolderPath):
        if f.startswith("boco.") or f.startswith("boco_chem."):
            # Copy file
            targetPath = NMMBEnvironment.UMO_OUT + f
            if not FileManagement.copyFile(os.path.abspath(f), targetPath):
                raise TaskExecutionException("[ERROR] Error copying file from " + f + " to " + targetPath)

    chemicRegSrc = NMMBEnvironment.OUTPUT + "chemic-reg"
    chemicRegTarget = NMMBEnvironment.UMO_OUT + "chemic-reg"
    if not FileManagement.copyFile(chemicRegSrc, chemicRegTarget):
        # TODO: Really no error when file does not exist?
        LOGGER_UMO_MODEL.debug("Cannot copy file from " + chemicRegSrc + " to " + chemicRegTarget + ". Skipping...")

    gwdSrc = NMMBEnvironment.OUTPUT + "GWD.bin"
    gwdTarget = NMMBEnvironment.UMO_OUT + "GWD.bin"
    if not FileManagement.copyFile(gwdSrc, gwdTarget):
        # TODO: Really no error when file does not exist?
        LOGGER_UMO_MODEL.debug("Cannot copy file from " + gwdSrc + " to " + gwdTarget + ". Skipping...")

    # Retrieve fcst from allPrep task
    inputDomain1Target = NMMBEnvironment.UMO_OUT + "input_domain_01"
    with open(inputDomain1Target, 'w') as dest:
        orig = compss_open(fcst)
        dest.write(orig.read())
        orig.close()

    # Retrieve soildust from dust_start task
    inputDomain2Target = NMMBEnvironment.UMO_OUT + "main_input_filename2"
    with open(inputDomain2Target, 'w') as dest:
        orig = compss_open(soildust)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh00 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH00Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh00"
    lookupAerosol2RH00Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH00"
    with open(lookupAerosol2RH00Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH00Src)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh50 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH50Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh50"
    lookupAerosol2RH50Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH50"
    with open(lookupAerosol2RH50Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH50Src)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh70 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH70Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh70"
    lookupAerosol2RH70Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH70"
    with open(lookupAerosol2RH70Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH70Src)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh80 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH80Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh80"
    lookupAerosol2RH80Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH80"
    with open(lookupAerosol2RH80Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH80Src)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh90 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH90Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh90"
    lookupAerosol2RH90Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH90"
    with open(lookupAerosol2RH90Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH90Src)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh95 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH95Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh95"
    lookupAerosol2RH95Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH95"
    with open(lookupAerosol2RH95Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH95Src)
        dest.write(orig.read())
        orig.close()

    # Retrieve lookup_aerosol2.dat.rh99 from run_aerosol   # TODO: Could be retrieved only once
    lookupAerosol2RH99Src = NMMBEnvironment.OUTPUT + "lookup_aerosol2.dat.rh99"
    lookupAerosol2RH99Target = NMMBEnvironment.UMO_OUT + "ETAMPNEW_AERO_RH99"
    with open(lookupAerosol2RH99Target, 'w') as dest:
        orig = compss_open(lookupAerosol2RH99Src)
        dest.write(orig.read())
        orig.close()

    # Copy coupling previous day (if required)
    if nmmbParams.getCoupleDustInit() and currentDate != nmmbParams.getStartDate():
        historySrc = NMMBEnvironment.OUTNMMB + nmmbParams.getCase() + os.path.sep + "history_INIT.hhh"
        historyTarget = NMMBEnvironment.UMO_OUT + "history_INIT.hhh"
        if not FileManagement.copyFile(historySrc, historyTarget):
            raise TaskExecutionException("[ERROR] Error copying file from " + historySrc + " to " + historyTarget)


def doUMOModel(nmmbParams, currentDate, fcst, soildust):
    """
    Performs the UMO Model simulation step
    :param nmmbParams: Nmmb parameters
    :param currentDate: Current date (datetime)
    :param fcst: Fcst
    :param soildust: Soil dust
    :raise TaskExecutionException: When the nems task fails or its preprocess or postprocess.
    """
    LOGGER_UMO_MODEL.info("Enter UMO Model process")

    # Prepare execution ********************************************************
    copyFilesFromPreprocess(nmmbParams, currentDate, fcst, soildust)
    nmmbParams.prepareUMOMOdelExecution(currentDate)
    umoModelMP = MessagePrinter(LOGGER_UMO_MODEL)

    # Begin MPI call ***********************************************************
    umoModelMP.printHeaderMsg("BEGIN")
    umoModelMP.printInfoMsg("Executing nmmb_esmf.x UMO-NMMb-DUST-RRTM model")

    stdOutFile = NMMBEnvironment.UMO_OUT + "nems.out"
    stdErrFile = NMMBEnvironment.UMO_OUT + "nems.err"
    nemsEV = MPI.nems(stdOutFile, stdErrFile)

    nemsEV = compss_wait_on(nemsEV)
    LOGGER_UMO_MODEL.debug("Execution of mpirun NEMS ended with status " + str(nemsEV))
    if nemsEV != 0:
        raise TaskExecutionException("[ERROR] Error executing mpirun nems")
    umoModelMP.printInfoMsg("Finished Executing nmmb_esmf.x UMO-NMMb-DUST-RRTM model")

    # Post execution ***********************************************************
    nmmbParams.postUMOModelExecution(currentDate)

    umoModelMP.printHeaderMsg("END")

    LOGGER_UMO_MODEL.info("UMO Model process finished")


def doPost(nmmbParams, currentDate):
    """
    Performs the POST step
    :param nmmbParams: Nmmb parameters
    :param currentDate: Current date (datetime)
    :raise TaskExecutionException: When the post tasks fail.
    :return; List of result files from the simulations
    """
    # Define model output folder by case and date
    currentDateSTR = currentDate.strftime(NMMBConstants.STR_TO_DATE)
    if nmmbParams.getHour() < 10:
        hourSTR = "0" + str(nmmbParams.getHour())
    else:
        hourSTR = str(nmmbParams.getHour())
    folderOutput = NMMBEnvironment.OUTNMMB + nmmbParams.getCase() + os.path.sep + currentDateSTR + hourSTR + os.path.sep

    LOGGER_POST.info("Postproc_carbono process for DAY: " + currentDateSTR)

    # Prepare execution ********************************************************
    nmmbParams.preparePostProcessExecution(currentDate)
    postProcMP = MessagePrinter(LOGGER_POST)

    # Deploy files and compile binaries if needed
    evCompile = BINARY.preparePost(nmmbParams.isCompileBinaries(), folderOutput)

    evCompile = compss_wait_on(evCompile)
    if evCompile != 0:
        raise TaskExecutionException("[ERROR] Error preparing post process")

    # Begin POST call **********************************************************
    postProcMP.printHeaderMsg("BEGIN")

    if nmmbParams.getDomain():
        domainSTR = "glob"
    else:
        domainSTR = "reg"
    dateHour = currentDateSTR + hourSTR
    pout = folderOutput + "new_pout_*.nc"
    CTM = folderOutput + "NMMB-BSC-CTM_" + dateHour + "_" + domainSTR + ".nc"

    ev = BINARY.executePostprocAuth(folderOutput, pout, CTM)

    ev = compss_wait_on(ev)
    LOGGER_POST.debug("Execution of NCRAT ended with status " + str(ev))
    if ev != 0:
        raise TaskExecutionException("[ERROR] Error executing post process")

    # Post execution ***********************************************************
    nmmbParams.cleanPostProcessExecution(folderOutput)

    postProcMP.printHeaderMsg("END")

    LOGGER_POST.info("Post process finished")

    return CTM


def doImages(nmmbParams, simulation_results):
    """
    Launches the images creation
    :param nmmbParams: Nmmb parameters
    :param simulation_results: Simulation result files
    :return: Dictionary containing the references to the images
    """
    LOGGER_IMAGE.info("Initiating images creation")
    # Define vars to be generated
    vars = ['acprec',
            'alwtoa',
            'dust_aod_550',
            'dust_aod_550_b1',
            'dust_aod_550_b2',
            'dust_aod_550_b3',
            'dust_aod_550_b4',
            'dust_aod_550_b5',
            'dust_aod_550_b6',
            'dust_aod_550_b7',
            'dust_aod_550_b8',
            'dust_drydep',
            'dust_load',
            'dust_load_b1',
            'dust_load_b2',
            'dust_load_b3',
            'dust_load_b4',
            'dust_load_b5',
            'dust_load_b6',
            'dust_load_b7',
            'dust_load_b8',
            'dust_pm10_sconc10',
            'dust_pm25_sconc10',
            'dust_sconc',
            'dust_sconc02',
            'dust_sconc10',
            'dust_sconc_b1',
            'dust_sconc_b2',
            'dust_sconc_b3',
            'dust_sconc_b4',
            'dust_sconc_b5',
            'dust_sconc_b6',
            'dust_sconc_b7',
            'dust_sconc_b8',
            'dust_wetdep',
            'dust_wetdep_cuprec',
            'fis',
            'ps',
            'slp',
            'u10',
            'v10']
    '''
    vars = ['dust_drydep',
            'dust_load',
            'dust_sconc10',
            'dust_wetdep']
    '''
    LOGGER_IMAGE.info("Images to create: " + str(vars))
    LOGGER_IMAGE.info("Days to process: " + str(len(simulation_results)))
    # Define images output folder
    folderOutput = NMMBEnvironment.OUTNMMB + nmmbParams.getCase()
    figures_folder = folderOutput + os.path.sep + 'figures'
    LOGGER_IMAGE.info("Images destination folder: " + figures_folder)

    # If figures folder already exists, do not remove previous. Create new one instead.
    if os.path.exists(figures_folder) and os.path.isdir(figures_folder):
        figures_folder = figures_folder + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        LOGGER_IMAGE.info("Images folder already exists, creating a new one in " + figures_folder)
    try:
        os.mkdir(figures_folder)
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error

    # Iterate over the results and generate the figures
    imgs = {}
    for v in vars:
        LOGGER_IMAGE.info("Creating images for variable: " + v)
        var_folder = figures_folder + os.path.sep + v
        os.mkdir(var_folder)
        imgs[v] = []
        i = 0
        for d in simulation_results:
            LOGGER_IMAGE.info(" - Simulation result: " + str(d))
            folder_date = os.path.basename(os.path.dirname(d))
            dt = datetime.datetime.strptime(folder_date[:-2], '%Y%m%d')
            date = dt.strftime('%Y-%m-%d')
            i1 = var_folder + os.path.sep + str(i) + '_0.png'
            i2 = var_folder + os.path.sep + str(i) + '_1.png'
            i3 = var_folder + os.path.sep + str(i) + '_2.png'
            i4 = var_folder + os.path.sep + str(i) + '_3.png'
            i5 = var_folder + os.path.sep + str(i) + '_4.png'
            i6 = var_folder + os.path.sep + str(i) + '_5.png'
            i7 = var_folder + os.path.sep + str(i) + '_6.png'
            i8 = var_folder + os.path.sep + str(i) + '_7.png'
            i9 = var_folder + os.path.sep + str(i) + '_8.png'
            imgs[v] += [i1, i2, i3, i4, i5, i6, i7, i8, i9]
            generate_figures(date, d, v, i1, i2, i3, i4, i5, i6, i7, i8, i9)
            i += 1

    LOGGER_IMAGE.info("Finished images creation submission")
    return imgs


def doAnimations(nmmbParams, imgs, skip_frames0=True):
    """
    Creates an animation for the given image references
    :param imgs: Dictionary containing the vars and a list of files where the snapshots are
    :param skip_frames0: Skip frame 0 of each iteration (some of them are empty and maybe repeated (24 - 00 ?)
    """
    LOGGER_ANIMATION.info("Initiating animation creation")
    folderOutput = NMMBEnvironment.OUTNMMB + nmmbParams.getCase()

    # Define images output folder
    animations_folder = folderOutput + os.path.sep + 'animations'
    LOGGER_IMAGE.info("Animations destination folder: " + animations_folder)
    # If figures folder already exists, do not remove previous. Create new one instead.
    if os.path.exists(animations_folder) and os.path.isdir(animations_folder):
        animations_folder = animations_folder + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        LOGGER_IMAGE.info("Animations folder already exists, creating a new one in " + animations_folder)
    try:
        os.mkdir(animations_folder)
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error

    animations = {}
    for var, imglist in imgs.iteritems():
        LOGGER_IMAGE.info("Creating animations for variable: " + var)
        LOGGER_IMAGE.info(" - Files: " + str(imglist))
        animation = animations_folder + os.path.sep + var + '.gif'
        animations[var] = animation
        generate_animation(animation, skip_frames0, *imglist)

    LOGGER_ANIMATION.info("Finished animation creation")


def main():
    """
    Main NMMB Workflow
    sys.args[0] : Configuration file path.
    :raise MainExecutionException: When something goes really wrong.
    """
    LOGGER_MAIN.info("Starting NMMB application")
    startTime = time.time()

    # Check and get arguments
    if len(sys.argv) != 2:
        usage()
        exit(1)

    configurationFile = sys.argv[1]

    # Load
    nmmbConfigManager = None
    try:
        LOGGER_MAIN.info("Loading NMMB Configuration file " + configurationFile)
        nmmbConfigManager = NMMBConfigManager(configurationFile)
        LOGGER_MAIN.info("Configuration file loaded")
    except:
        LOGGER_MAIN.error("[ERROR] Cannot load configuration file: " + configurationFile + ". Aborting...")
        raise MainExecutionException("Cannot load configuration file: " + configurationFile)

    # Compute the execution variables NMMBParameters
    nmmbParams = NMMBParameters(nmmbConfigManager)

    # Prepare the execution
    nmmbParams.prepareExecution()

    # Fixed process(do before main time looping)
    startFixed = time.time()
    if nmmbParams.doFixed():
        try:
            doFixed(nmmbParams)
        except TaskExecutionException as tee:
            LOGGER_FIXED.error("[ERROR] Task exception on fixed phase. Aborting...")
            raise MainExecutionException(tee)
        finally:
            endFixed = time.time()
            LOGGER_FIXED.info("[TIME] FIXED START   = " + str(startFixed))
            LOGGER_FIXED.info("[TIME] FIXED END     = " + str(endFixed))
            LOGGER_FIXED.info("[TIME] FIXED ELAPSED = " + str(endFixed - startFixed))

    # Start main time loop
    currentDate = nmmbParams.getStartDate()
    simulation_results = []
    while currentDate <= nmmbParams.getEndDate():
        currentDateSTR = currentDate.strftime(NMMBConstants.STR_TO_DATE)
        LOGGER_MAIN.info(currentDateSTR + " simulation started")

        # Create output folders if needed
        nmmbParams.createOutputFolders(currentDate)

        # Vrbl process
        startVariable = time.time()
        if nmmbParams.doVariable():
            try:
                fcst, soildust = doVariable(nmmbParams, currentDate)
            except TaskExecutionException as tee:
                LOGGER_VARIABLE.error("[ERROR] Task exception on variable phase at date " + str(currentDateSTR) + ". Aborting...")
                raise MainExecutionException(tee)
            finally:
                endVariable = time.time()
                LOGGER_VARIABLE.info("[TIME] VARIABLE START   = " + str(startVariable))
                LOGGER_VARIABLE.info("[TIME] VARIABLE END     = " + str(endVariable))
                LOGGER_VARIABLE.info("[TIME] VARIABLE ELAPSED = " + str(endVariable - startVariable))

        # UMO model run
        startUMO = time.time()
        if nmmbParams.doUmoModel():
            try:
                doUMOModel(nmmbParams, currentDate, fcst, soildust)
            except TaskExecutionException as tee:
                LOGGER_UMO_MODEL.error("[ERROR] Task exception on UMO Model phase at date " + str(currentDateSTR) + ". Aborting...")
                raise MainExecutionException(tee)
            finally:
                endUMO = time.time()
                LOGGER_UMO_MODEL.info("[TIME] UMO START   = " + str(startUMO))
                LOGGER_UMO_MODEL.info("[TIME] UMO END     = " + str(endUMO))
                LOGGER_UMO_MODEL.info("[TIME] UMO ELAPSED = " + str(endUMO - startUMO))

        # Post process
        startPost = time.time()
        if nmmbParams.doPost():
            try:
                simulation_results.append(doPost(nmmbParams, currentDate))
            except TaskExecutionException as tee:
                LOGGER_POST.error("[ERROR] Task exception on Post phase at date " + str(currentDateSTR) + ". Aborting...")
                raise MainExecutionException(tee)
            finally:
                endPost = time.time()
                LOGGER_POST.info("[TIME] POST START   = " + str(startPost))
                LOGGER_POST.info("[TIME] POST END     = " + str(endPost))
                LOGGER_POST.info("[TIME] POST ELAPSED = " + str(endPost - startPost))

        LOGGER_MAIN.info(currentDateSTR + " simulation finished")

        # Getting next simulation day
        currentDate = currentDate + datetime.timedelta(seconds=NMMBConstants.ONE_DAY_IN_SECONDS)

    # Image creation
    startImagesCreation = time.time()
    imgs = doImages(nmmbParams, simulation_results)
    endImagesCreation = time.time()
    LOGGER_IMAGE.info("[TIME] IMAGES START   = " + str(startImagesCreation))
    LOGGER_IMAGE.info("[TIME] IMAGES END     = " + str(endImagesCreation))
    LOGGER_IMAGE.info("[TIME] IMAGES ELAPSED = " + str(endImagesCreation - startImagesCreation))

    # Animation creation
    startAnimationsCreation = time.time()
    doAnimations(nmmbParams, imgs, True)  # Skip frame 0 due to empty snapshot (usually)
    endAnimationsCreation = time.time()
    LOGGER_IMAGE.info("[TIME] ANIMATIONS START   = " + str(startAnimationsCreation))
    LOGGER_IMAGE.info("[TIME] ANIMATIONS END     = " + str(endAnimationsCreation))
    LOGGER_IMAGE.info("[TIME] ANIMATIONS ELAPSED = " + str(endAnimationsCreation - startAnimationsCreation))

    compss_barrier()

    # Print execution time
    endTime = time.time()
    LOGGER_MAIN.info("[TIME] TOTAL START   = " + str(startTime))
    LOGGER_MAIN.info("[TIME] TOTAL END     = " + str(endTime))
    LOGGER_MAIN.info("[TIME] TOTAL ELAPSED = " + str(endTime - startTime))


if __name__ == '__main__':
    main()
