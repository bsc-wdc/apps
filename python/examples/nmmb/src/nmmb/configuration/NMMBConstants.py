"""
Class to represent the constant names and environment variable names for NMMB
"""

# -----------------------------------------------------------------------
# Workflow modifications
CLEAN_BINARIES_NAME = "CLEAN_BINARIES"
COMPILE_BINARIES_NAME = "COMPILE_BINARIES"

# -----------------------------------------------------------------------
# MN settings
INPES_NAME = "INPES"
JNPES_NAME = "JNPES"
WRTSK_NAME = "WRTSK"

# -----------------------------------------------------------------------
# Global-regional switch - Model domain setup global/regional
DOMAIN_NAME = "DOMAIN"
LM_NAME = "LM"
CASE_NAME = "CASE"

# -----------------------------------------------------------------------
# If regional you need to modify manually files llgrid_chem.inc in vrbl409rrtm_bsc1.0_reg
DT_INT1_NAME = "DT_INT1"
TLM0D1_NAME = "TLM0D1"
TPH0D1_NAME = "TPH0D1"
WBD1_NAME = "WBD1"
SBD1_NAME = "SBD1"
DLMD1_NAME = "DLMD1"
DPHD1_NAME = "DPHD1"
PTOP1_NAME = "PTOP1"
DCAL1_NAME = "DCAL1"
NRADS1_NAME = "NRADS1"
NRADL1_NAME = "NRADL1"
# -----------------------------------------------------------------------
DT_INT2_NAME = "DT_INT2"
TLM0D2_NAME = "TLM0D2"
TPH0D2_NAME = "TPH0D2"
WBD2_NAME = "WBD2"
SBD2_NAME = "SBD2"
DLMD2_NAME = "DLMD2"
DPHD2_NAME = "DPHD2"
PTOP2_NAME = "PTOP2"
DCAL2_NAME = "DCAL2"
NRADS2_NAME = "NRADS2"
NRADL2_NAME = "NRADL2"

# -----------------------------------------------------------------------
# Case selection
DO_FIXED_NAME = "DO_FIXED"
DO_VRBL_NAME = "DO_VRBL"
DO_UMO_NAME = "DO_UMO"
DO_POST_NAME = "DO_POST"

# -----------------------------------------------------------------------
# Select START and ENDING Times
START_DATE_NAME = "START"
END_DATE_NAME = "END"
HOUR_NAME = "HOUR"
NHOURS_NAME = "NHOURS"
NHOURS_INIT_NAME = "NHOURS"
HIST_NAME = "HIST"
BOCO_NAME = "BOCO"
TYPE_GFSINIT_NAME = "TYPE_GFSINIT"
TYPE_GFSINIT_FNL = "FNL"

# -----------------------------------------------------------------------
# Select configuration of POSTPROC (DO_POST)
HOUR_P_NAME = "HOUR_P"
NHOURS_P_NAME = "NHOURS_P"
HIST_P_NAME = "HIST_P"
LSM_NAME = "LSM"

# -----------------------------------------------------------------------
# Select IC of chemistry for run with COUPLE_DUST_INIT=0
INIT_CHEM_NAME = "INIT_CHEM"

# -----------------------------------------------------------------------
# Couple dust
COUPLE_DUST_NAME = "COUPLE_DUST"
COUPLE_DUST_INIT_NAME = "COUPLE_DUST_INIT"

# -----------------------------------------------------------------------
# Environment variable names
ENV_NAME_UMO_ROOT = "UMO_ROOT"
ENV_NAME_UMO_PATH = "UMO_PATH"
ENV_NAME_FIX = "FIX"
ENV_NAME_VRB = "VRB"
ENV_NAME_OUTPUT = "OUTPUT"
ENV_NAME_UMO_OUT = "UMO_OUT"
ENV_NAME_OUTNMMB = "OUTNMMB"
ENV_NAME_FNL = "FNL"
ENV_NAME_GFS = "GFS"
ENV_NAME_SRCDIR = "SRCDIR"
ENV_NAME_CHEMIC = "CHEMIC"
ENV_NAME_DATMOD = "DATMOD"
ENV_NAME_POST_CARBONO = "POST_CARBONO"

# -----------------------------------------------------------------------
# Format conversion constants
STR_TO_DATE = '%Y%m%d'
COMPACT_STR_TO_DATE = '%y%m%d'
MONTH_NAME_DATE_TO_STR = '%d%m%Y'
DATE_TO_YEAR = '%Y'
DATE_TO_MONTH = '%m'
DATE_TO_DAY = '%d'
ONE_DAY_IN_SECONDS = 1 * 24 * 60 * 60
HOUR_TO_MINUTES = 60
