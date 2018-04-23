import os
import logging
import errno
from datetime import datetime

import NMMBConstants
import NMMBEnvironment

from userexceptions.CommandException import CommandException
from loggers import LoggerNames
from utils.BashCMDExecutor import BashCMDExecutor
import utils.FileManagement as FileManagement
import utils.FortranWrapper as FortranWrapper


class NMMBParameters (object):
    """
    Representation of the parameters of a NMMB execution
    """

    # Loggers
    LOGGER_MAIN = logging.getLogger(LoggerNames.NMMB_MAIN)
    LOGGER_FIXED = logging.getLogger(LoggerNames.NMMB_FIXED)
    LOGGER_VARIABLE = logging.getLogger(LoggerNames.NMMB_VARIABLE)
    LOGGER_UMO_MODEL = logging.getLogger(LoggerNames.NMMB_UMO_MODEL)
    LOGGER_POST = logging.getLogger(LoggerNames.NMMB_POST)

    # -----------------------------------------------------------------------
    # Load workflow modifications
    CLEAN_BINARIES = None
    COMPILE_BINARIES = None

    # -----------------------------------------------------------------------
    # MN settings
    INPES = None
    JNPES = None
    WRTSK = None
    PROC = None

    # -----------------------------------------------------------------------
    # Global-regional switch - Model domain setup global/regional
    DOMAIN = None
    LM = None
    CASE = None

    # -----------------------------------------------------------------------
    # Model variables
    DT_INT = None
    TLM0D = None
    TPH0D = None
    WBD = None
    SBD = None
    DLMD = None
    DPHD = None
    PTOP = None
    DCAL = None
    NRADS = None
    NRADL = None
    IMI = None
    JMI = None
    IM = None
    JM = None

    # -----------------------------------------------------------------------
    # Case selection
    DO_FIXED = None
    DO_VRBL = None
    DO_UMO = None
    DO_POST = None

    # -----------------------------------------------------------------------
    # Select START and ENDING Times
    START_DATE = None
    END_DATE = None
    HOUR = None
    NHOURS = None
    NHOURS_INIT = None
    HIST = None
    HIST_M = None
    BOCO = None
    TYPE_GFSINIT = None

    # -----------------------------------------------------------------------
    # Select configuration of POSTPROC (DO_POST)
    HOUR_P = None
    NHOURS_P = None
    HIST_P = None
    LSM = None
    TPH0DN = None
    WBDDEF = None
    SBDDEF = None

    # -----------------------------------------------------------------------
    # Select IC of chemistry for run with COUPLE_DUST_INIT=0
    INIT_CHEM = None

    # -----------------------------------------------------------------------
    # Couple dust
    COUPLE_DUST = None
    COUPLE_DUST_INIT = None

    def __init__(self, nmmbConfiguration):
        """
        Constructor
        :param nmmbConfiguration: Nmmb configuration
        """
        self.LOGGER_MAIN.info("Setting execution variables...")

        # Load workflow modifications
        self.CLEAN_BINARIES = nmmbConfiguration.getCleanBinaries()
        self.COMPILE_BINARIES = nmmbConfiguration.getCompileBinaries()

        # MN settings
        self.INPES = nmmbConfiguration.getINPES()
        self.JNPES = nmmbConfiguration.getJNPES()
        self.WRTSK = nmmbConfiguration.getWRTSK()
        self.PROC = self.INPES * self.JNPES + self.WRTSK

        # Global-regional switch - Model domain setup global/regional
        self.DOMAIN = nmmbConfiguration.getDomain()
        self.LM = nmmbConfiguration.getLM()
        self.CASE = nmmbConfiguration.getCase()

        # Model variables
        self.DT_INT = nmmbConfiguration.getDT_INT1() if self.DOMAIN else nmmbConfiguration.getDT_INT2()
        self.TLM0D = nmmbConfiguration.getTLM0D1() if self.DOMAIN else nmmbConfiguration.getTLM0D2()
        self.TPH0D = nmmbConfiguration.getTPH0D1() if self.DOMAIN else nmmbConfiguration.getTPH0D2()
        self.WBD = nmmbConfiguration.getWBD1() if self.DOMAIN else nmmbConfiguration.getWBD2()
        self.SBD = nmmbConfiguration.getSBD1() if self.DOMAIN else nmmbConfiguration.getSBD2()
        self.DLMD = nmmbConfiguration.getDLMD1() if self.DOMAIN else nmmbConfiguration.getDLMD2()
        self.DPHD = nmmbConfiguration.getDPHD1() if self.DOMAIN else nmmbConfiguration.getDPHD2()
        self.PTOP = nmmbConfiguration.getPTOP1() if self.DOMAIN else nmmbConfiguration.getPTOP2()
        self.DCAL = nmmbConfiguration.getDCAL1() if self.DOMAIN else nmmbConfiguration.getDCAL2()
        self.NRADS = nmmbConfiguration.getNRADS1() if self.DOMAIN else nmmbConfiguration.getNRADS2()
        self.NRADL = nmmbConfiguration.getNRADL1() if self.DOMAIN else nmmbConfiguration.getNRADL2()
        self.IMI = int(-2.0 * self.WBD / self.DLMD + 1.5)
        self.JMI = int(-2.0 * self.SBD / self.DPHD + 1.5)
        self.IM = self.IMI + 2 if self.DOMAIN else self.IMI
        self.JM = self.JMI + 2 if self.DOMAIN else self.JMI

        self.LOGGER_MAIN.info("")
        self.LOGGER_MAIN.info("Number of processors " + str(self.PROC))
        self.LOGGER_MAIN.info("Model grid size - IM / JM / LM: " + str(self.IMI) + " / " + str(self.JMI) + " / " + str(self.LM))
        self.LOGGER_MAIN.info("Extended domain - IM / JM / LM: " + str(self.IM) + " / " + str(self.JM) + " / " + str(self.LM))
        self.LOGGER_MAIN.info("")

        # Case selection
        self.DO_FIXED = nmmbConfiguration.getFixed()
        self.DO_VRBL = nmmbConfiguration.getVariable()
        self.DO_UMO = nmmbConfiguration.getUmoModel()
        self.DO_POST = nmmbConfiguration.getPost()

        # -----------------------------------------------------------------------
        # Select START and ENDING Times
        sd = None   # Date
        try:
            sd = datetime.strptime(nmmbConfiguration.getStartDate(), NMMBConstants.STR_TO_DATE)
        except:
            self.LOGGER_MAIN.error("[ERROR] Cannot parse start date")
            self.LOGGER_MAIN.error("Aborting...")
            exit(1)
        finally:
            self.START_DATE = sd

        ed = None  # Date
        try:
            ed = datetime.strptime(nmmbConfiguration.getEndDate(), NMMBConstants.STR_TO_DATE)
        except:
            self.LOGGER_MAIN.error("[ERROR] Cannot parse end date")
            self.LOGGER_MAIN.error("Aborting...")
            exit(1)
        finally:
            self.END_DATE = ed

        self.HOUR = nmmbConfiguration.getHour()
        self.NHOURS = nmmbConfiguration.getNHours()
        self.NHOURS_INIT = nmmbConfiguration.getNHoursInit()
        self.HIST = nmmbConfiguration.getHist()
        self.HIST_M = self.HIST * NMMBConstants.HOUR_TO_MINUTES
        self.BOCO = nmmbConfiguration.getBoco()
        self.TYPE_GFSINIT = nmmbConfiguration.getTypeGFSInit()

        # -----------------------------------------------------------------------
        # Select configuration of POSTPROC (DO_POST)
        self.HOUR_P = nmmbConfiguration.getHourP()
        self.NHOURS_P = nmmbConfiguration.getNHoursP()
        self.HIST_P = nmmbConfiguration.getHistP()
        self.LSM = nmmbConfiguration.getLSM()

        self.TPH0DN = nmmbConfiguration.getTPH0D2() + 90.0
        self.WBDDEF = nmmbConfiguration.getWBD2() + nmmbConfiguration.getTLM0D2()
        self.SBDDEF = nmmbConfiguration.getSBD2() + nmmbConfiguration.getTPH0D2()

        # -----------------------------------------------------------------------
        # Select IC of chemistry for run with COUPLE_DUST_INIT=0
        self.INIT_CHEM = nmmbConfiguration.getInitChem()
        self.COUPLE_DUST = nmmbConfiguration.getCoupleDust()
        self.COUPLE_DUST_INIT = nmmbConfiguration.getCoupleDustInit()

        self.LOGGER_MAIN.info("Execution variables set")

    def isCleanBinaries(self):
        """
        Returns if binaries must be erased or not
        :return: self CLEAN_BINARIES value
        """
        return self.CLEAN_BINARIES

    def isCompileBinaries(self):
        """
        Returns if binaries must be compiled or not
        :return: self COMPILE_BINARIES value
        """
        return self.COMPILE_BINARIES

    def getDomain(self):
        """
        Returns the DOMAIN value
        :return: self DOMAIN value
        """
        return self.DOMAIN

    def getCase(self):
        """
        Returns the CASE value
        :return: self CASE value
        """
        return self.CASE

    def doFixed(self):
        """
        Returns the DO_FIXED value
        :return: self DO_FIXED value
        """
        return self.DO_FIXED

    def doVariable(self):
        """
        Returns the DO_VARIABLE value
        :return: self DO_VARIABLE value
        """
        return self.DO_VRBL

    def doUmoModel(self):
        """
        Returns the DO_UMO value
        :return: self DO_UMO value
        """
        return self.DO_UMO

    def doPost(self):
        """
        Returns the DO_POST value
        :return: self DO_POST value
        """
        return self.DO_POST

    def getStartDate(self):
        """
        Returns the START_DATE
        :return: self START_DATE value
        """
        return self.START_DATE

    def getEndDate(self):
        """
        Returns the END_DATE
        :return: self END_DATE value
        """
        return self.END_DATE

    def getHour(self):
        """
        Returns the HOUR value
        :return: self HOUR value
        """
        return self.HOUR

    def getCoupleDustInit(self):
        """
        Returns the COUPLE_DUST_INIT value
        :return: self COUPLE_DUST_INIT value
        """
        return self.COUPLE_DUST_INIT

    def prepareExecution(self):
        """
        Actions to perform to setup an NMMB execution
        """
        self.LOGGER_MAIN.info("Preparing execution...")

        # Define folders
        outputPath = NMMBEnvironment.UMO_OUT
        outputCasePath = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep + "output" + os.path.sep
        outputSymPath = NMMBEnvironment.UMO_PATH + "PREPROC" + os.path.sep + "output"

        # Clean folders
        self.LOGGER_MAIN.debug("Clean output folder : " + outputPath)
        FileManagement.deleteFileOrFolder(outputPath)
        self.LOGGER_MAIN.debug("Clean output folder : " + outputCasePath)
        FileManagement.deleteFileOrFolder(outputCasePath)
        self.LOGGER_MAIN.debug("Clean output folder : " + outputSymPath)
        FileManagement.deleteFileOrFolder(outputSymPath)

        # Create empty files
        self.LOGGER_MAIN.debug("Create output folder : " + outputPath)
        if not FileManagement.createDir(outputPath):
            self.LOGGER_MAIN.error("[ERROR] Cannot create output folder")
            self.LOGGER_MAIN.error("Aborting...")
            exit(1)
        self.LOGGER_MAIN.debug("Create output folder : " + outputCasePath)
        if not FileManagement.createDir(outputCasePath):
            self.LOGGER_MAIN.error("[ERROR] Cannot create output case folder")
            self.LOGGER_MAIN.error("Aborting...")
            exit(1)

        # Symbolic link for preprocess
        self.LOGGER_MAIN.debug("Symbolic link for PREPROC output folder")
        self.LOGGER_MAIN.debug("   - From: " + outputCasePath)
        self.LOGGER_MAIN.debug("   - To:   " + outputSymPath)
        try:
            os.symlink(outputCasePath, outputSymPath)
        except Exception as e:
            self.LOGGER_MAIN.error("[ERROR] Cannot create output symlink ", e)
            self.LOGGER_MAIN.error("Aborting...")
            exit(1)

        self.LOGGER_MAIN.info("Execution environment prepared")

    '''
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ******************** FIXED STEP *******************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    '''

    def prepareFixedExecution(self):
        """
        Actions to setup the FIXED Step of an NMMB execution
        """
        self.LOGGER_FIXED.debug("   - INCLUDE PATH : " + NMMBEnvironment.FIX_INCLUDE_DIR)

        modelgridTMPFilePath = NMMBEnvironment.FIX_INCLUDE_DIR + "modelgrid_rrtm.tmp"
        lmimjmTMPFilePath = NMMBEnvironment.FIX_INCLUDE_DIR + "lmimjm_rrtm.tmp"
        modelgridFilePath = NMMBEnvironment.FIX_INCLUDE_DIR + "modelgrid.inc"
        lmimjmFilePath = NMMBEnvironment.FIX_INCLUDE_DIR + "lmimjm.inc"

        # Clean some files
        self.LOGGER_FIXED.debug("Delete previous: " + modelgridFilePath)
        if not FileManagement.deleteFile(modelgridFilePath):
            self.LOGGER_FIXED.debug("Cannot erase previous modelgrid because it doesn't exist.")
        self.LOGGER_FIXED.debug("Delete previous: " + lmimjmFilePath)
        if not FileManagement.deleteFile(lmimjmFilePath):
            self.LOGGER_FIXED.debug("Cannot erase previous lmimjm because it doesn't exist.")

        # Prepare files
        cmdModelgrid = BashCMDExecutor("sed")
        cmdModelgrid.addFlagAndValue("-e", "s/TLMD/" + str(self.TLM0D) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/TPHD/" + str(self.TPH0D) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/WBDN/" + str(self.WBD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/SBDN/" + str(self.SBD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/DLMN/" + str(self.DLMD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/DPHN/" + str(self.DPHD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/IBDY/" + str(self.IM) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/JBDY/" + str(self.JM) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/PTOP/" + str(self.PTOP) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/KKK/" + str(self.LM) + "/")
        cmdModelgrid.addArgument(modelgridTMPFilePath)
        cmdModelgrid.redirectOutput(modelgridFilePath)
        try:
            ev = cmdModelgrid.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_FIXED.error("[ERROR] Error performing sed command on model grid " + modelgridTMPFilePath, ce)
            self.LOGGER_FIXED.error("Aborting...")
            exit(1)

        cmdLmimjm = BashCMDExecutor("sed")
        cmdLmimjm.addFlagAndValue("-e", "s/TLMD/" + str(self.TLM0D) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/TPHD/" + str(self.TPH0D) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/WBDN/" + str(self.WBD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/SBDN/" + str(self.SBD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/DLMN/" + str(self.DLMD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/DPHN/" + str(self.DPHD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/IBDY/" + str(self.IM) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/JBDY/" + str(self.JM) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/PTOP/" + str(self.PTOP) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/KKK/" + str(self.LM) + "/")
        cmdLmimjm.addArgument(lmimjmTMPFilePath)
        cmdLmimjm.redirectOutput(lmimjmFilePath)
        try:
            ev = cmdLmimjm.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_FIXED.error("[ERROR] Error performing sed command on Lmimjm " + lmimjmTMPFilePath, ce)
            self.LOGGER_FIXED.error("Aborting...")
            exit(1)

    '''
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ******************** OUTPUT FOLDERS ***************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    '''

    def createOutputFolders(self, currentDate):
        """
        Actions to create the output folder for a date iteration of an NMMB execution
        :param currentDate: Current date (datetime)
        """
        currentDateSTR = currentDate.strftime(NMMBConstants.STR_TO_DATE)
        hourSTR = "0" + str(self.HOUR) if self.HOUR < 10 else str(self.HOUR)
        folderOutputCase = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep
        folderOutput = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep + currentDateSTR + hourSTR + os.path.sep

        if not FileManagement.createDir(folderOutputCase):
            self.LOGGER_MAIN.debug("Cannot create folder output case : " + folderOutputCase + " because it already exists. Skipping")

        if not FileManagement.createDir(folderOutput):
            self.LOGGER_MAIN.debug("Cannot create folder output : " + folderOutput + " because it already exists. Skipping")

    '''
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ******************** VARIABLE STEP ****************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    '''

    def prepareVariableExecution(self, currentDate):
        """
        Actions to prepare the VARIABLE Step of an NMMB Execution
        :param currentDate: Current date (datetime)
        """
        # Clean specific files
        outputFiles = ["sst2dvar_grb_0.5", "fcst", "llstmp", "llsmst", "llgsno", "llgcic", "llgsst", "llspl.000",
                       "llgsst05", "albedo", "albase", "vegfrac", "z0base", "z0", "ustar", "sst05", "dzsoil", "tskin",
                       "sst", "snow", "snowheight", "cice", "seamaskcorr", "landusecorr", "landusenewcorr",
                       "topsoiltypecorr", "vegfraccorr", "z0corr", "z0basecorr", "emissivity", "canopywater",
                       "frozenprecratio", "smst", "sh2o", "stmp"]

        for outf in outputFiles:
            filePath = NMMBEnvironment.OUTPUT + outf
            if not FileManagement.deleteFile(filePath):
                self.LOGGER_VARIABLE.debug("Cannot erase previous " + outf + " because it doesn't exist.")

        # Clean regular expr files
        folder = NMMBEnvironment.OUTPUT
        for fn in os.listdir(folder):
            if fn.endswith(".gfs") or fn.startswith("gfs.") or fn.startswith("boco.") or fn.startswith("boco_chem."):
                if not FileManagement.deleteFile(fn):
                    self.LOGGER_VARIABLE.debug("Cannot erase previous " + fn + " because it doesn't exist.")

        # Clean files on VRB
        sstgrbFilePath = NMMBEnvironment.VRB + "sstgrb"
        if not FileManagement.deleteFile(sstgrbFilePath):
            self.LOGGER_VARIABLE.debug("Cannot erase previous sstgrb because it doesn't exist.")

        llgridFilePath = NMMBEnvironment.VRB_INCLUDE_DIR + "llgrid.inc"
        if not FileManagement.deleteFile(llgridFilePath):
            self.LOGGER_VARIABLE.debug("Cannot erase previous llgrid.inc because it doesn't exist.")

        # Prepare files
        fullDate = currentDate.strftime(NMMBConstants.STR_TO_DATE)
        compactDate = currentDate.strftime(NMMBConstants.COMPACT_STR_TO_DATE)
        hourSTR = "0" + str(self.HOUR) if self.HOUR < 10 else str(self.HOUR)
        nHoursSTR = "0" + str(self.NHOURS) if self.NHOURS < 10 else str(self.NHOURS)

        llgridSrcFile = NMMBEnvironment.VRB_INCLUDE_DIR + "llgrid_rrtm_" + self.TYPE_GFSINIT + ".tmp"
        llgridFile = NMMBEnvironment.VRB_INCLUDE_DIR + "llgrid.inc"
        cmdllgrid = BashCMDExecutor("sed")
        cmdllgrid.addFlagAndValue("-e", "s/LLL/" + nHoursSTR + "/")
        cmdllgrid.addFlagAndValue("-e", "s/HH/" + hourSTR + "/")
        cmdllgrid.addFlagAndValue("-e", "s/UPBD/" + str(self.BOCO) + "/")
        cmdllgrid.addFlagAndValue("-e", "s/YYYYMMDD/" + str(compactDate) + "/")
        cmdllgrid.addArgument(llgridSrcFile)
        cmdllgrid.redirectOutput(llgridFile)
        try:
            ev = cmdllgrid.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_VARIABLE.error("[ERROR] Error performing sed command on model grid " + llgridSrcFile, ce)
            self.LOGGER_VARIABLE.error("Aborting...")
            exit(1)

        if self.DOMAIN:
            if self.TYPE_GFSINIT == NMMBConstants.TYPE_GFSINIT_FNL:
                try:
                    target = NMMBEnvironment.FNL + "fnl_" + fullDate + "_" + hourSTR + "_00"
                    link = NMMBEnvironment.OUTPUT + "gfs.t" + hourSTR + "z.pgrbf00"
                    if not FileManagement.deleteFile(link):
                        self.LOGGER_VARIABLE.debug("Cannot erase previous link " + link + " because it doesn't exist.")
                    os.symlink(target, link)
                    self.LOGGER_VARIABLE.debug("Softlink from : " + link + " to " + target)
                except Exception as e:
                    self.LOGGER_VARIABLE.error("[ERROR] Cannot create output symlink", e)
                    self.LOGGER_VARIABLE.error("Aborting...")
                    exit(1)
            else:
                self.LOGGER_VARIABLE.info("Converting wafs.00.0P5DEG from grib2 to grib1")
                input = NMMBEnvironment.GFS + "wafs.00.0P5DEG"
                output = NMMBEnvironment.OUTPUT + "gfs.t" + hourSTR + "z.pgrbf00"

                cnvgrib = BashCMDExecutor("cnvgrib")
                cnvgrib.addArgument("-g21")
                cnvgrib.addArgument(input)
                cnvgrib.addArgument(output)
                try:
                    ev = cnvgrib.execute()
                    if ev != 0:
                        raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
                except CommandException as ce:
                    self.LOGGER_VARIABLE.error("[ERROR] Error performing cnvgrib command", ce)
                    self.LOGGER_VARIABLE.error("Aborting...")
                    exit(1)
        else:
            # Domain is 1
            if self.TYPE_GFSINIT == NMMBConstants.TYPE_GFSINIT_FNL:
                for i in range(self.HOUR, self.BOCO, self.NHOURS):  # for (int i=self.HOUR; i<self.BOCO; i+=self.NHOURS)
                    try:
                        iStr = "0" + str(i) if i < 10 else str(i)
                        dDay = i / 24
                        hDay = i % 24
                        hDayStr = "0" + str(hDay) if hDay < 10 else str(hDay)
                        newDay = currentDate + datetime.timedelta(seconds=dDay * NMMBConstants.ONE_DAY_IN_SECONDS)
                        dayDateStr = newDay.strftime(NMMBConstants.ONE_DAY_IN_SECONDS)

                        target = NMMBEnvironment.FNL + "fnl_" + dayDateStr + "_" + hDayStr + "_00"
                        link = NMMBEnvironment.OUTPUT + "gfs.t" + hourSTR + "z.pgrbf" + iStr
                        if not FileManagement.deleteFile(link):
                            self.LOGGER_VARIABLE.debug("Cannot erase previous link " + link + " because it doesn't exist.")
                        os.symlink(target, link)
                        self.LOGGER_VARIABLE.debug("Softlink from : " + link + " to " + target)
                    except Exception as e:
                        self.LOGGER_VARIABLE.error("[ERROR] Cannot create output symlink", e)
                        self.LOGGER_VARIABLE.error("Aborting...")
                        exit(1)
            else:
                for i in range(0, self.BOCO, self.NHOURS):  # for (int i = 0 i < self.BOCO i += self.NHOURS) {
                    iStr = "0" + str(i) if i < 10 else str(i)
                    input = NMMBEnvironment.GFS + "wafs." + iStr + ".0P5DEG"
                    output = NMMBEnvironment.OUTPUT + "gfs.t" + hourSTR + "z.pgrbf" + iStr

                    cnvgrib = BashCMDExecutor("cnvgrib")
                    cnvgrib.addArgument("-g21")
                    cnvgrib.addArgument(input)
                    cnvgrib.addArgument(output)
                    try:
                        ev = cnvgrib.execute()
                        if ev != 0:
                            raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
                    except CommandException as ce:
                        self.LOGGER_VARIABLE.error("[ERROR] Error performing cnvgrib command", ce)
                        self.LOGGER_VARIABLE.error("Aborting...")
                        exit(1)

        # Prepare modelgrid and lmimjm files
        modelgridTMPFilePath = NMMBEnvironment.VRB_INCLUDE_DIR + "modelgrid_rrtm.tmp"
        lmimjmTMPFilePath = NMMBEnvironment.VRB_INCLUDE_DIR + "lmimjm_rrtm.tmp"
        modelgridFilePath = NMMBEnvironment.VRB_INCLUDE_DIR + "modelgrid.inc"
        lmimjmFilePath = NMMBEnvironment.VRB_INCLUDE_DIR + "lmimjm.inc"

        # Clean some files
        self.LOGGER_VARIABLE.debug("Delete previous: " + modelgridFilePath)
        if not FileManagement.deleteFile(modelgridFilePath):
            self.LOGGER_VARIABLE.debug("Cannot erase previous modelgrid because it doesn't exist.")
        self.LOGGER_VARIABLE.debug("Delete previous: " + lmimjmFilePath)
        if not FileManagement.deleteFile(lmimjmFilePath):
            self.LOGGER_VARIABLE.debug("Cannot erase previous lmimjm because it doesn't exist.")

        # Prepare files
        cmdModelgrid = BashCMDExecutor("sed")
        cmdModelgrid.addFlagAndValue("-e", "s/TLMD/" + str(self.TLM0D) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/TPHD/" + str(self.TPH0D) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/WBDN/" + str(self.WBD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/SBDN/" + str(self.SBD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/DLMN/" + str(self.DLMD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/DPHN/" + str(self.DPHD) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/IBDY/" + str(self.IM) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/JBDY/" + str(self.JM) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/PTOP/" + str(self.PTOP) + "/")
        cmdModelgrid.addFlagAndValue("-e", "s/KKK/" + str(self.LM) + "/")
        cmdModelgrid.addArgument(modelgridTMPFilePath)
        cmdModelgrid.redirectOutput(modelgridFilePath)
        try:
            ev = cmdModelgrid.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_VARIABLE.error("[ERROR] Error performing sed command on model grid " + modelgridTMPFilePath, ce)
            self.LOGGER_VARIABLE.error("Aborting...")
            exit(1)

        cmdLmimjm = BashCMDExecutor("sed")
        cmdLmimjm.addFlagAndValue("-e", "s/TLMD/" + str(self.TLM0D) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/TPHD/" + str(self.TPH0D) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/WBDN/" + str(self.WBD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/SBDN/" + str(self.SBD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/DLMN/" + str(self.DLMD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/DPHN/" + str(self.DPHD) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/IBDY/" + str(self.IM) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/JBDY/" + str(self.JM) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/PTOP/" + str(self.PTOP) + "/")
        cmdLmimjm.addFlagAndValue("-e", "s/KKK/" + str(self.LM) + "/")
        cmdLmimjm.addArgument(lmimjmTMPFilePath)
        cmdLmimjm.redirectOutput(lmimjmFilePath)
        try:
            ev = cmdLmimjm.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_VARIABLE.error("[ERROR] Error performing sed command on Lmimjm " + lmimjmTMPFilePath, ce)
            self.LOGGER_VARIABLE.error("Aborting...")
            exit(1)

    def postVariableExecution(self, targetFolder):
        srcFilePath = NMMBEnvironment.VRB_INCLUDE_DIR + "lmimjm.inc"
        targetFilePath = targetFolder + "lmimjm.inc"
        if not FileManagement.copyFile(srcFilePath, targetFilePath):
            self.LOGGER_VARIABLE.error("[ERROR] Error copying lmimjm.inc file to " + targetFolder)
            self.LOGGER_VARIABLE.error("Aborting...")
            exit(1)

    '''
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ******************** UMO MODEL STEP ***************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    '''

    def prepareUMOMOdelExecution(self, currentDate):
        """
        Actions to prepare the UMO Model Execution Step of an NMMB Execution
        :param currentDate: Current date (datetime)
        """
        # Copy data files
        if NMMBEnvironment.CHEMIC is None or NMMBEnvironment.CHEMIC == '':
            self.LOGGER_UMO_MODEL.debug("Cannot copy from CHEMIC because source doesn't exist. Skipping...")
        else:
            dataFolder = NMMBEnvironment.CHEMIC + "MEGAN" + os.path.sep + "out" + os.path.sep + "aqmeii-reg" + os.path.sep
            chemicFound = False
            try:
                contentFiles = os.listdir(dataFolder)
                chemicFound = True
            except:
                self.LOGGER_UMO_MODEL.debug("Failed to find CHEMIC source folder because it doesn't exist.")
            if chemicFound and len(contentFiles) > 0:
                for f in contentFiles:
                    if f == "isop.dat" or (f.startswith("lai") and f.endswith(".dat")) or f == "meteo-data.dat" or (f.startswith("pftp_") and f.endswith(".dat")):
                        # Copy file
                        targetPath = NMMBEnvironment.UMO_OUT + f
                        if not FileManagement.copyFile(dataFolder + f, targetPath):
                            self.LOGGER_UMO_MODEL.debug("Cannot copy " + f + " file to " + targetPath + " because source doesn't exist. Skipping...")
            else:
                self.LOGGER_UMO_MODEL.debug("Cannot copy from CHEMIC because source doesn't exist. Skipping...")

        # Dust coupling part 1
        coupleDustIteration = False
        if currentDate > self.START_DATE:
            coupleDustIteration = self.COUPLE_DUST

            if self.COUPLE_DUST_INIT:
                coupleDustIteration = True

        dustFlag = "EEEE/true" if coupleDustIteration else "EEEE/false"

        # Prepare config rrtm chem file
        nHoursSTR = "0" + str(self.NHOURS) if self.NHOURS < 10 else str(self.NHOURS)
        yearSTR = currentDate.strftime(NMMBConstants.DATE_TO_YEAR)
        monthSTR = currentDate.strftime(NMMBConstants.DATE_TO_MONTH)
        daySTR = currentDate.strftime(NMMBConstants.DATE_TO_DAY)
        hourSTR = "0" + str(self.HOUR) if self.HOUR < 10 else str(self.HOUR)

        configFileTMPPath = NMMBEnvironment.UMO_ROOT + "configfile_rrtm_chem.tmp"
        configFilePath = NMMBEnvironment.UMO_OUT + "configure_file"
        configFile = BashCMDExecutor("sed")
        configFile.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
        configFile.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
        configFile.addFlagAndValue("-e", "s/KKK/" + str(self.LM) + "/")
        configFile.addFlagAndValue("-e", "s/TPHD/" + str(self.TPH0D) + "/")
        configFile.addFlagAndValue("-e", "s/TLMD/" + str(self.TLM0D) + "/")
        configFile.addFlagAndValue("-e", "s/WBD/" + str(self.WBD) + "/")
        configFile.addFlagAndValue("-e", "s/SBD/" + str(self.SBD) + "/")
        configFile.addFlagAndValue("-e", "s/INPES/" + str(self.INPES) + "/")
        configFile.addFlagAndValue("-e", "s/JNPES/" + str(self.JNPES) + "/")
        configFile.addFlagAndValue("-e", "s/WRTSK/" + str(self.WRTSK) + "/")
        configFile.addFlagAndValue("-e", "s/DTINT/" + str(self.DT_INT) + "/")
        configFile.addFlagAndValue("-e", "s/YYYY/" + yearSTR + "/")
        configFile.addFlagAndValue("-e", "s/MM/" + monthSTR + "/")
        configFile.addFlagAndValue("-e", "s/DD/" + daySTR + "/")
        configFile.addFlagAndValue("-e", "s/HH/" + hourSTR + "/")
        configFile.addFlagAndValue("-e", "s/LLL/" + nHoursSTR + "/")
        configFile.addFlagAndValue("-e", "s/STT/" + str(self.HIST_M) + "/")
        configFile.addFlagAndValue("-e", "s/DOM/true/")
        configFile.addFlagAndValue("-e", "s/" + dustFlag + "/")
        configFile.addFlagAndValue("-e", "s/BBBB/" + str(self.DCAL) + "/")
        configFile.addFlagAndValue("-e", "s/NRADS/" + str(self.NRADS) + "/")
        configFile.addFlagAndValue("-e", "s/NRADL/" + str(self.NRADL) + "/")
        configFile.addFlagAndValue("-e", "s/CCCC/" + str(self.INIT_CHEM) + "/")

        configFile.addArgument(configFileTMPPath)
        configFile.redirectOutput(configFilePath)
        try:
            ev = configFile.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_VARIABLE.error("[ERROR] Error performing sed command on configFile " + configFileTMPPath, ce)
            self.LOGGER_VARIABLE.error("Aborting...")
            exit(1)

        # Copy datmod
        datModFolder = NMMBEnvironment.DATMOD
        for f in os.listdir(datModFolder):
            # Copy all files
            source = datModFolder + os.path.sep + f
            if os.path.isfile(source):
                targetPath = NMMBEnvironment.UMO_OUT + f
                if not FileManagement.copyFile(source, targetPath):
                    self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + source + " file to " + targetPath)
                    self.LOGGER_UMO_MODEL.error("Aborting...")
                    exit(1)

        lookupDatSrc = NMMBEnvironment.DATMOD + "nam_micro_lookup.dat"
        lookupDatTarget = NMMBEnvironment.UMO_OUT + "ETAMPNEW_DATA"
        if not FileManagement.copyFile(lookupDatSrc, lookupDatTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + lookupDatSrc + " file to " + lookupDatTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        wrftablesFolder = NMMBEnvironment.DATMOD + "wrftables" + os.path.sep
        for f in os.listdir(wrftablesFolder):
            # Copy all files
            source = wrftablesFolder + os.path.sep + f
            if os.path.isfile(source):
                targetPath = NMMBEnvironment.UMO_OUT + f
                if not FileManagement.copyFile(source, targetPath):
                    self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + source + " file to " + targetPath)
                    self.LOGGER_UMO_MODEL.error("Aborting...")
                    exit(1)

        co2dataFolder = NMMBEnvironment.DATMOD + "co2data" + os.path.sep
        for f in os.listdir(co2dataFolder):
            # Copy all files
            source = co2dataFolder + os.path.sep + f
            if os.path.isfile(source):
                targetPath = NMMBEnvironment.UMO_OUT + f
                if not FileManagement.copyFile(source, targetPath):
                    self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + source + " file to " + targetPath)
                    self.LOGGER_UMO_MODEL.error("Aborting...")
                    exit(1)

        # Copy files for RRTM radiation
        climaGlobalSrc = NMMBEnvironment.DATMOD + "fix" + os.path.sep + "fix_rad" + os.path.sep + "global_climaeropac_global.txt"
        climaGlobalTarget = NMMBEnvironment.UMO_OUT + "aerosol.dat"
        if not FileManagement.copyFile(climaGlobalSrc, climaGlobalTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + climaGlobalSrc + " file to " + climaGlobalTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        solarSrc = NMMBEnvironment.DATMOD + "fix" + os.path.sep + "fix_rad" + os.path.sep + "solarconstantdata.txt"
        solarTarget = NMMBEnvironment.UMO_OUT + "solarconstantdata.txt"
        if not FileManagement.copyFile(solarSrc, solarTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + solarSrc + " file to " + solarTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        fixRadFolder = NMMBEnvironment.DATMOD + "fix" + os.path.sep + "fix_rad" + os.path.sep
        for f in os.listdir(fixRadFolder):
            source = fixRadFolder + os.path.sep + f
            if os.path.isfile(source) and (f.startswith("co2historicaldata") or f.startswith("volcanic_aerosols_")):
                targetPath = NMMBEnvironment.UMO_OUT + f
                if not FileManagement.copyFile(source, targetPath):
                    self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + source + " file to " + targetPath)
                    self.LOGGER_UMO_MODEL.error("Aborting...")
                    exit(1)

        # Copy files for GoCart climatology conc. and opt. properties
        fixGocartFolder = NMMBEnvironment.DATMOD + "fix" + os.path.sep + "fix_gocart_clim" + os.path.sep
        for f in os.listdir(fixGocartFolder):
            source = fixGocartFolder + os.path.sep + f
            if os.path.isfile(source) and f.startswith("2000"):
                targetPath = NMMBEnvironment.UMO_OUT + f
                if not FileManagement.copyFile(source, targetPath):
                    self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + source + " file to " + targetPath)
                    self.LOGGER_UMO_MODEL.error("Aborting...")
                    exit(1)

        ncepAerosolSrc = NMMBEnvironment.DATMOD + "fix" + os.path.sep + "fix_aeropt_luts" + os.path.sep + "NCEP_AEROSOL.bin"
        ncepAerosolTarget = NMMBEnvironment.UMO_OUT + "NCEP_AEROSOL.bin"
        if not FileManagement.copyFile(ncepAerosolSrc, ncepAerosolTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + ncepAerosolSrc + " file to " + ncepAerosolTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        # Copy files for chemistry tests
        # TODO: Emissions copy file from /gpfs/bsc32/BLA/NNMB/RUN/FUKU-DATA/xe133_emissions.dat
        # emissionsSrc = "xe133_emissions.dat"
        # emissionsTarget = NMMBEnvironment.UMO_OUT + "xe133_emissions.dat"
        # if not FileManagement.copyFile(emissionsSrc, emissionsTarget):
        #     self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + emissionsSrc + " file to " + emissionsTarget)
        #     self.LOGGER_UMO_MODEL.error("Aborting...")
        #     exit(1)

        configure01Src = NMMBEnvironment.UMO_OUT + "configure_file"
        configure01Target = NMMBEnvironment.UMO_OUT + "configure_file_01"
        if not FileManagement.copyFile(configure01Src, configure01Target):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + configure01Src + " file to " + configure01Target)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        modelConfigureSrc = NMMBEnvironment.UMO_OUT + "configure_file"
        modelConfigureTarget = NMMBEnvironment.UMO_OUT + "model_configure"
        if not FileManagement.copyFile(modelConfigureSrc, modelConfigureTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + modelConfigureSrc + " file to " + modelConfigureTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        solverSrc = NMMBEnvironment.SRCDIR + "NAMELISTS" + os.path.sep + "solver_state.txt"
        solverTarget = NMMBEnvironment.UMO_OUT + "solver_state.txt"
        if not FileManagement.copyFile(solverSrc, solverTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + solverSrc + " file to " + solverTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        oceanSrc = NMMBEnvironment.SRCDIR + "NAMELISTS" + os.path.sep + "ocean.configure"
        oceanTarget = NMMBEnvironment.UMO_OUT + "ocean.configure"
        if not FileManagement.copyFile(oceanSrc, oceanTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + oceanSrc + " file to " + oceanTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        atmosSrc = NMMBEnvironment.SRCDIR + "NAMELISTS" + os.path.sep + "atmos.configure"
        atmosTarget = NMMBEnvironment.UMO_OUT + "atmos.configure"
        if not FileManagement.copyFile(atmosSrc, atmosTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + atmosSrc + " file to " + atmosTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        globalPrdlosSrc = NMMBEnvironment.DATMOD + "global_o3prdlos.f77"
        globalPrdlosTarget = NMMBEnvironment.UMO_OUT + "fort.28"
        try:
            os.symlink(globalPrdlosSrc, globalPrdlosTarget)
        except OSError, e:
            if e.errno == errno.EEXIST:
                self.LOGGER_UMO_MODEL.debug("[DEBUG] The symlink " + globalPrdlosTarget + " from " + globalPrdlosSrc + " already exists.")

        globalClimSrc = NMMBEnvironment.DATMOD + "global_o3clim.txt"
        globalClimTarget = NMMBEnvironment.UMO_OUT + "fort.48"
        try:
            os.symlink(globalClimSrc, globalClimTarget)
        except OSError, e:
            if e.errno == errno.EEXIST:
                self.LOGGER_UMO_MODEL.debug("[DEBUG] The symlink " + globalClimTarget + " from " + globalClimSrc + " already exists.")

    def postUMOModelExecution(self, currentDate):
        """
        Actions to perform after the UMO Model Execution
        :param currentDate: Current date (datetime)
        """
        # Define model output folder by case and date
        currentDateSTR = currentDate.strftime(NMMBConstants.STR_TO_DATE)
        hourSTR = "0" + str(self.HOUR) if self.HOUR < 10 else str(self.HOUR)
        folderOutputCase = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep
        folderOutput = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep + currentDateSTR + hourSTR + os.path.sep

        historyFilePath = NMMBEnvironment.UMO_OUT + "history_INIT.hhh"
        FileManagement.deleteFile(historyFilePath)

        if self.COUPLE_DUST:
            historyTarget = folderOutputCase + "history_INIT.hhh"
            if self.NHOURS_INIT < 100:
                if self.NHOURS_INIT < 10:
                    historySrc = NMMBEnvironment.UMO_OUT + "nmmb_hst_01_bin_000" + str(self.NHOURS_INIT) + "h_00m_00.00s"
                else:
                    historySrc = NMMBEnvironment.UMO_OUT + "nmmb_hst_01_bin_00" + str(self.NHOURS_INIT) + "h_00m_00.00s"
            else:
                historySrc = NMMBEnvironment.UMO_OUT + "nmmb_hst_01_bin_0" + str(self.NHOURS_INIT) + "h_00m_00.00s"

            if not FileManagement.copyFile(historySrc, historyTarget):
                self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + historySrc + " file to " + historyTarget)
                self.LOGGER_UMO_MODEL.error("Aborting...")
                exit(1)

        nmmRrtmOutSrc = NMMBEnvironment.UMO_OUT + "nmm_rrtm.out"
        nmmRrtmOutTarget = folderOutput + "nmm_rrtm.out"
        if not FileManagement.moveFile(nmmRrtmOutSrc, nmmRrtmOutTarget):
            # TODO: We don't really need to abort when cannot copy self file?
            self.LOGGER_UMO_MODEL.error("Cannot copy " + nmmRrtmOutSrc + " file to " + nmmRrtmOutTarget)

        configureFileSrc = NMMBEnvironment.UMO_OUT + "configure_file"
        configureFileTarget = folderOutput + "configure_file"
        if not FileManagement.moveFile(configureFileSrc, configureFileTarget):
            self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + configureFileSrc + " file to " + configureFileTarget)
            self.LOGGER_UMO_MODEL.error("Aborting...")
            exit(1)

        boundarySrc = NMMBEnvironment.OUTPUT + "boundary_ecmwf.nc"
        boundaryTarget = folderOutput + "boundary_ecmwf.nc"
        if not FileManagement.moveFile(boundarySrc, boundaryTarget):
            # TODO: We don't really need to abort when cannot copy self file?
            self.LOGGER_UMO_MODEL.warn("Cannot copy " + boundarySrc + " file to " + boundaryTarget)

        umoFolder = NMMBEnvironment.UMO_OUT
        for f in os.listdir(umoFolder):
            source = umoFolder + f
            if os.path.isfile(source) and f.startswith("nmmb_hst_01_bin_"):
                target = folderOutput + f
                if not FileManagement.moveFile(source, target):
                    self.LOGGER_UMO_MODEL.error("[ERROR] Error copying " + source + " file to " + target)
                    self.LOGGER_UMO_MODEL.error("Aborting...")
                    exit(1)

    '''
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ******************** POST PROCESS STEP ************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    * ***************************************************************************************************
    '''

    def preparePostProcessExecution(self, currentDate):
        """
        Actions to perform on the POST PROCESS Step of an NMMB Execution
        :param currentDate: Current date (datetime)
        """
        # Define model output folder by case and date
        currentDateSTR = currentDate.strftime(NMMBConstants.STR_TO_DATE)
        hourSTR = "0" + str(self.HOUR) if self.HOUR < 10 else str(self.HOUR)
        folderOutputCase = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep
        folderOutput = NMMBEnvironment.OUTNMMB + self.CASE + os.path.sep + currentDateSTR + hourSTR + os.path.sep

        lmimjmSrc = folderOutputCase + "lmimjm.inc"
        lmimjmTarget = NMMBEnvironment.POST_CARBONO + "lmimjm.inc"
        if not FileManagement.copyFile(lmimjmSrc, lmimjmTarget):
            self.LOGGER_POST.error("[ERROR] Error copying " + lmimjmSrc + " file to " + lmimjmTarget)
            self.LOGGER_POST.error("Aborting...")
            exit(1)

        # Clean new_postall.f if needed
        postAllTarget = NMMBEnvironment.POST_CARBONO + "new_postall.f"
        self.LOGGER_POST.debug("Delete previous: " + postAllTarget)
        if not FileManagement.deleteFile(postAllTarget):
            self.LOGGER_POST.debug("Cannot erase previous new_postall.f because it doesn't exist.")

        # Create new new_postall.f content
        postAllSrc = NMMBEnvironment.POST_CARBONO + "new_postall.f.tmp"
        hourPSTR = "0" + str(self.HOUR_P) if self.HOUR_P < 10 else str(self.HOUR_P)
        nHoursPSTR = "0" + str(self.NHOURS_P) if self.NHOURS_P < 10 else str(self.NHOURS_P)
        cmdPostall = BashCMDExecutor("sed")
        cmdPostall.addFlagAndValue("-e", "s/QQQ/" + nHoursPSTR + "/")
        cmdPostall.addFlagAndValue("-e", "s/SSS/" + hourPSTR + "/")
        cmdPostall.addFlagAndValue("-e", "s/TTT/" + str(self.HIST_P) + "/")
        cmdPostall.addArgument(postAllSrc)
        cmdPostall.redirectOutput(postAllTarget)
        try:
            ev = cmdPostall.execute()
            if ev != 0:
                raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
        except CommandException as ce:
            self.LOGGER_POST.error("[ERROR] Error performing sed command on " + postAllSrc, ce)
            self.LOGGER_POST.error("Aborting...")
            exit(1)

        datePost = currentDate.strftime(NMMBConstants.MONTH_NAME_DATE_TO_STR)
        tdeft = int(self.NHOURS_P / (self.HIST + 1))
        tdef = "0" + str(tdeft) if tdeft < 10 else str(tdeft)

        if self.DOMAIN:
            poutGlobalCtlSrc = NMMBEnvironment.POST_CARBONO + "pout_global_pressure.ctl.tmp"
            poutGlobalCtlTarget = folderOutput + "pout_global_pressure_" + currentDateSTR + hourSTR + ".ctl"
            cmdGlobalCtl = BashCMDExecutor("sed")
            cmdGlobalCtl.addFlagAndValue("-e", "s/DATE/" + currentDateSTR + hourSTR + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/WBDN/" + str(self.WBD) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/DLMN/" + str(self.DLMD) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/SBDN/" + str(self.SBD) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/DPHN/" + str(self.DPHD) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/KKK/" + str(self.LSM) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/HH/" + tdef + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/INITCTL/" + str(self.HOUR) + "Z" + str(datePost) + "/")
            cmdGlobalCtl.addFlagAndValue("-e", "s/XHR/" + str(self.HIST_P) + "hr/")
            cmdGlobalCtl.addArgument(poutGlobalCtlSrc)
            cmdGlobalCtl.redirectOutput(poutGlobalCtlTarget)
            try:
                ev = cmdGlobalCtl.execute()
                if ev != 0:
                    raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
            except CommandException as ce:
                self.LOGGER_POST.error("[ERROR] Error performing sed command on " + poutGlobalCtlSrc, ce)
                self.LOGGER_POST.error("Aborting...")
                exit(1)
        else:
            ireg = self.IMI - 2
            jreg = self.JMI - 2
            poutRegionalCtlSrc = NMMBEnvironment.POST_CARBONO + "pout_regional_pressure.ctl.tmp"
            poutRegionalCtlTarget = folderOutput + "pout_regional_pressure_" + currentDateSTR + hourSTR + ".ctl"
            cmdRegionalCtl = BashCMDExecutor("sed")
            cmdRegionalCtl.addFlagAndValue("-e", "s/DATE/" + currentDateSTR + hourSTR + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/IRG/" + str(ireg) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/JRG/" + str(jreg) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/TLMN/" + str(self.TLM0D) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/TPHN/" + str(self.TPH0DN) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/DLMN/" + str(self.DLMD) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/DPHN/" + str(self.DPHD) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/WBDN/" + str(self.WBD) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/SBDN/" + str(self.SBD) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/III/" + str(self.IMI) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/JJJ/" + str(self.JMI) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/WBXX/" + str(self.WBDDEF) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/SBYY/" + str(self.SBDDEF) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/KKK/" + str(self.LSM) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/HH/" + tdef + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/INITCTL/" + str(self.HOUR) + "Z" + str(datePost) + "/")
            cmdRegionalCtl.addFlagAndValue("-e", "s/XHR/" + str(self.HIST_P) + "hr/")
            cmdRegionalCtl.addArgument(poutRegionalCtlSrc)
            cmdRegionalCtl.redirectOutput(poutRegionalCtlTarget)
            try:
                ev = cmdRegionalCtl.execute()
                if ev != 0:
                    raise CommandException("[ERROR] CMD returned non-zero exit value: " + ev)
            except CommandException as ce:
                self.LOGGER_POST.error("[ERROR] Error performing sed command on " + poutRegionalCtlSrc, ce)
                self.LOGGER_POST.error("Aborting...")
                exit(1)

    def cleanPostProcessExecution(self, folderOutput):
        """
        Clean post process execution files
        :param folderOutput: Output folder to clean
        """
        # Clean files
        namelist = folderOutput + "namelist.newpost"
        if os.path.exists(namelist):
            os.remove(namelist)

        source = folderOutput + FortranWrapper.NEW_POSTALL + FortranWrapper.SUFFIX_F_SRC
        if os.path.exists(source):
            os.remove(source)

        lmimjm = folderOutput + "lmimjm.inc"
        if os.path.exists(lmimjm):
            os.remove(lmimjm)

        executable = folderOutput + FortranWrapper.NEW_POSTALL + FortranWrapper.SUFFIX_EXE
        if os.path.exists(executable):
            os.remove(executable)

        if self.CLEAN_BINARIES:
            preExecutable = NMMBEnvironment.POST_CARBONO + FortranWrapper.NEW_POSTALL + FortranWrapper.SUFFIX_EXE
            if os.path.exists(preExecutable):
                os.remove(preExecutable)
