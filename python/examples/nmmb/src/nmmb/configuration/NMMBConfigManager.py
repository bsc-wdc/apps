import ConfigParser
import StringIO
import NMMBConstants


class NMMBConfigManager(object):
    """
    Loads the NMMB configuration
    """

    mainSection = 'Root, config'  # required for the configparser
    config = ConfigParser.RawConfigParser()

    def __init__(self, pathToConfigFile):
        """
        Loads the runtime configuration found in path pathToConfigFile.
        :param pathToConfigFile: Path to configuration file.
        """
        ini_str = '[' + self.mainSection + ']\n' + open(pathToConfigFile, 'r').read()
        ini_fp = StringIO.StringIO(ini_str)
        self.config.readfp(ini_fp)

    def getCleanBinaries(self):
        """
        Returns the CLEAN_BINARIES value
        :return: CLEAN_BINARIES value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.CLEAN_BINARIES_NAME)

    def getCompileBinaries(self):
        """
        Returns the COMPILE_BINARIES value
        :return: COMPILE_BINARIES value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.COMPILE_BINARIES_NAME)

    def getINPES(self):
        """
        Returns the INPES value
        :return: INPES value
        """
        return self.config.getint(self.mainSection, NMMBConstants.INPES_NAME)

    def getJNPES(self):
        """
        Returns the JNPES value
        :return: JNPES value
        """
        return self.config.getint(self.mainSection, NMMBConstants.JNPES_NAME)

    def getWRTSK(self):
        """
        Returns the WRTSK value
        :return: WRTSK value
        """
        return self.config.getint(self.mainSection, NMMBConstants.WRTSK_NAME)

    def getDomain(self):
        """
        Returns the DOMAIN value
        :return: DOMAIN value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.DOMAIN_NAME)

    def getLM(self):
        """
        Returns the LM value
        :return: LM value
        """
        return self.config.getint(self.mainSection, NMMBConstants.LM_NAME)

    def getCase(self):
        """
        Returns the CASE name value
        :return: CASE name value
        """
        return self.config.get(self.mainSection, NMMBConstants.CASE_NAME)

    def getDT_INT1(self):
        """
        Returns the DT_INT1 value
        :return: DT_INT1 value
        """
        return self.config.getint(self.mainSection, NMMBConstants.DT_INT1_NAME)

    def getDT_INT2(self):
        """
        Returns the DT_INT2 value
        :return: DT_INT2 value
        """
        return self.config.getint(self.mainSection, NMMBConstants.DT_INT2_NAME)

    def getTLM0D1(self):
        """
        Returns the TLM0D1 value
        :return: TLM0D1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.TLM0D1_NAME)

    def getTLM0D2(self):
        """
        Returns the TLM0D2 value
        :return: TLM0D2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.TLM0D2_NAME)

    def getTPH0D1(self):
        """
        Returns the TPH0D1 value
        :return: TPH0D1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.TPH0D1_NAME)

    def getTPH0D2(self):
        """
        Returns the TPH0D2 value
        :return: TPH0D2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.TPH0D2_NAME)

    def getWBD1(self):
        """
        Returns the WBD1 value
        :return: WBD1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.WBD1_NAME)

    def getWBD2(self):
        """
        Returns the WBD2 value
        :return: WBD2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.WBD2_NAME)

    def getSBD1(self):
        """
        Returns the SBD1 value
        :return: SBD1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.SBD1_NAME)

    def getSBD2(self):
        """
        Returns the SBD2 value
        :return: SBD2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.SBD2_NAME)

    def getDLMD1(self):
        """
        Returns the DLMD1 value
        :return: DLMD1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.DLMD1_NAME)

    def getDLMD2(self):
        """
        Returns the DLMD2 value
        :return: DLMD2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.DLMD2_NAME)

    def getDPHD1(self):
        """
        Returns the DPHD1 value
        :return: DPHD1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.DPHD1_NAME)

    def getDPHD2(self):
        """
        Returns the DPHD2 value
        :return: DPHD2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.DPHD2_NAME)

    def getPTOP1(self):
        """
        Returns the PTOP1 value
        :return: PTOP1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.PTOP1_NAME)

    def getPTOP2(self):
        """
        Returns the PTOP2 value
        :return: PTOP2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.PTOP2_NAME)

    def getDCAL1(self):
        """
        Returns the DCAL1 value
        :return: DCAL1 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.DCAL1_NAME)

    def getDCAL2(self):
        """
        Returns the DCAL2 value
        :return: DCAL2 value
        """
        return self.config.getfloat(self.mainSection, NMMBConstants.DCAL2_NAME)

    def getNRADS1(self):
        """
        Returns the NRADS1 value
        :return: NRADS1 value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NRADS1_NAME)

    def getNRADS2(self):
        """
        Returns the NRADS2 value
        :return: NRADS2 value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NRADS2_NAME)

    def getNRADL1(self):
        """
        Returns the NRADL1 value
        :return: NRADL1 value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NRADL1_NAME)

    def getNRADL2(self):
        """
        Returns the NRADL2 value
        :return: NRADL2 value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NRADL2_NAME)

    def getFixed(self):
        """
        Returns the DO_FIXED value
        :return: DO_FIXED value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.DO_FIXED_NAME)

    def getVariable(self):
        """
        Returns the DO_VARIABLE value
        :return: DO_VARIABLE value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.DO_VRBL_NAME)

    def getUmoModel(self):
        """
        Returns the DO_UMO value
        :return: DO_UMO value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.DO_UMO_NAME)

    def getPost(self):
        """
        Returns the DO_POST value
        :return: DO_POST value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.DO_POST_NAME)

    def getStartDate(self):
        """
        Returns the START_DATE value
        :return: START_DATE value
        """
        return self.config.get(self.mainSection, NMMBConstants.START_DATE_NAME)

    def getEndDate(self):
        """
        Returns the END_DATE value
        :return: END_DATE value
        """
        return self.config.get(self.mainSection, NMMBConstants.END_DATE_NAME)

    def getInitChem(self):
        """
        Returns the INIT_CHEM value
        :return: INIT_CHEM value
        """
        return self.config.getint(self.mainSection, NMMBConstants.INIT_CHEM_NAME)

    def getCoupleDust(self):
        """
        Returns the COUPLE_DUST value
        :return: COUPLE_DUST value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.COUPLE_DUST_NAME)

    def getCoupleDustInit(self):
        """
        Returns the COUPLE_DUST_INIT value
        :return: COUPLE_DUST_INIT value
        """
        return self.config.getboolean(self.mainSection, NMMBConstants.COUPLE_DUST_INIT_NAME)

    def getHour(self):
        """
        Returns the HOUR value
        :return: HOUR value
        """
        return self.config.getint(self.mainSection, NMMBConstants.HOUR_NAME)

    def getNHours(self):
        """
        Returns the NHOURS value
        :return: NHOURS value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NHOURS_NAME)

    def getNHoursInit(self):
        """
        Returns the NHOURS_INIT value
        :return: NHOURS_INIT value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NHOURS_INIT_NAME)

    def getHist(self):
        """
        Returns the HIST value
        :return: HIST value
        """
        return self.config.getint(self.mainSection, NMMBConstants.HIST_NAME)

    def getBoco(self):
        """
        Returns the BOCO value
        :return: BOCO value
        """
        return self.config.getint(self.mainSection, NMMBConstants.BOCO_NAME)

    def getTypeGFSInit(self):
        """
        Returns the GSFINIT value
        :return: GSFINIT value
        """
        return self.config.get(self.mainSection, NMMBConstants.TYPE_GFSINIT_NAME)

    def getHourP(self):
        """
        Returns the HOUR_P value
        :return: HOUR_P value
        """
        return self.config.getint(self.mainSection, NMMBConstants.HOUR_P_NAME)

    def getNHoursP(self):
        """
        Returns the NHOURS_P value
        :return: NHOURS_P value
        """
        return self.config.getint(self.mainSection, NMMBConstants.NHOURS_P_NAME)

    def getHistP(self):
        """
        Returns the HIST_P value
        :return: HIST_P value
        """
        return self.config.getint(self.mainSection, NMMBConstants.HIST_P_NAME)

    def getLSM(self):
        """
        Returns the LSM value
        :return: LSM value
        """
        return self.config.getint(self.mainSection, NMMBConstants.LSM_NAME)
