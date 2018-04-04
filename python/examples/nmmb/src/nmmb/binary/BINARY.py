from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.binary import binary
import utils.FortranWrapper as FortranWrapper
import utils.BinaryWrapper as BinaryWrapper
import configuration.NMMBConstants as NMMBConstants


"""
* *****************************************************************************
* *****************************************************************************
* *****************************************************************************
* ******************** COMPILE STEP *******************************************
* *****************************************************************************
* *****************************************************************************
* *****************************************************************************
"""


@binary(binary=FortranWrapper.FC)
@task(returns=int, source=FILE_IN)
def fortranCompiler(mcFlag, sharedFlag, covertPrefix, convertValue,
                    tracebackFlag, assumePrefix, assumeValue, optFlag,
                    fpmodelPrefix, fpmodelValue, stackFlag, oFlag, executable,
                    source):
    pass


# source is a string here?
@binary(binary=FortranWrapper.FC)
@task(returns=int, object=FILE_OUT)
def fortranCompileObject(mcFlag, sharedFlag, covertPrefix, convertValue,
                         tracebackFlag, assumePrefix, assumeValue, optFlag,
                         fpmodelPrefix, fpmodelValue, stackFlag, cFlag, source,
                         oFlag, object, moduleFlag, moduleDir):
    pass


@binary(binary=FortranWrapper.FC)
@task(returns=int, object=FILE_IN)
def fortranCompileWithObject(mcFlag, sharedFlag, covertPrefix, convertValue,
                             tracebackFlag, assumePrefix, assumeValue, optFlag,
                             fpmodelPrefix, fpmodelValue, stackFlag, oFlag,
                             executable, source, object):
    pass


@binary(binary=FortranWrapper.FC)
@task(returns=int, source=FILE_IN)
def fortranCompilerWithW3(mcFlag, sharedFlag, covertPrefix, convertValue,
                          tracebackFlag, assumePrefix, assumeValue, optFlag,
                          fpmodelPrefix, fpmodelValue, stackFlag, oFlag,
                          executable, source, w3libDir, bacioLibDir, w3Lib,
                          bacioLib):
    pass


@binary(binary=FortranWrapper.GFC)
@task(returns=int, source=FILE_IN)
def gfortranCompiler(bigOFlag, source, oFlag, executable):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + BinaryWrapper.COMPILE_READ_PAUL_SOURCE)
@task(returns=int, source=FILE_IN)
def compileReadPaulSource(source, executable):
    pass


"""
* ****************************************************************************
* ****************************************************************************
* ****************************************************************************
* ******************** FIXED STEP ********************************************
* ****************************************************************************
* ****************************************************************************
* ****************************************************************************
"""


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.SMMOUNT + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamaskDEM=FILE_OUT, heightDEM=FILE_OUT)
def smmount(topoDir, seamaskDEM, heightDEM):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.LANDUSE + FortranWrapper.SUFFIX_EXE)
@task(returns=int, land_use=FILE_OUT, kount_landuse=FILE_OUT)
def landuse(landuseDir, land_use, kount_landuse):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.LANDUSENEW + FortranWrapper.SUFFIX_EXE)
@task(returns=int, land_use_new=FILE_OUT, kount_landusenew=FILE_OUT)
def landusenew(gtopDir, land_use_new, kount_landusenew):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.TOPO + FortranWrapper.SUFFIX_EXE)
@task(returns=int, heightmean=FILE_OUT)
def topo(topoDir, heightmean):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.STDH + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamaskDEM=FILE_IN, heightmean=FILE_IN, st_dh=FILE_OUT)
def stdh(seamaskDEM, heightmean, topoDir, st_dh):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.ENVELOPE + FortranWrapper.SUFFIX_EXE)
@task(returns=int, heightmean=FILE_IN, st_dh=FILE_IN, height=FILE_OUT)
def envelope(heightmean, st_dh, height):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.TOPSOILTYPE + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamaskDEM=FILE_IN, top_soil_type=FILE_OUT)
def topsoiltype(seamaskDEM, soiltypeDir, top_soil_type):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.BOTSOILTYPE + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamaskDEM=FILE_IN, bot_soil_type=FILE_OUT)
def botsoiltype(seamaskDEM, soiltypePath, bot_soil_type):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.TOPOSEAMASK + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamaskDEM=FILE_IN, seamask=FILE_OUT, height=FILE_INOUT, land_use=FILE_INOUT,
      top_soil_type=FILE_INOUT, bot_soil_type=FILE_INOUT)
def toposeamask(seamaskDEM, seamask, height, land_use, top_soil_type, bot_soil_type):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.STDHTOPO + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamask=FILE_IN, st_dh=FILE_INOUT)
def stdhtopo(seamask, st_dh):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.DEEPTEMPERATURE + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamask=FILE_IN, deep_temperature=FILE_OUT)
def deeptemperature(seamask, soiltempPath, deep_temperature):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.SNOWALBEDO + FortranWrapper.SUFFIX_EXE)
@task(returns=int, snow_albedo=FILE_OUT)
def snowalbedo(maxsnowalbDir, snow_albedo):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.VCGENERATOR + FortranWrapper.SUFFIX_EXE)
@task(returns=int, dsg=FILE_OUT)
def vcgenerator(dsg):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.ROUGHNESS + FortranWrapper.SUFFIX_EXE)
@task(returns=int, Roughness=FILE_OUT)
def roughness(roughnessDir, Roughness):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.GFDLCO2 + FortranWrapper.SUFFIX_EXE)
@task(returns=int, dsg=FILE_IN, co2_trans=FILE_OUT)
def gfdlco2(dsg, co2_data_dir, co2_trans):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_FIX + "}/lookup_tables/" + BinaryWrapper.RUN_AEROSOL)
@task(returns=int, lookup_aerosol2_rh00=FILE_OUT, lookup_aerosol2_rh50=FILE_OUT, lookup_aerosol2_rh70=FILE_OUT,
      lookup_aerosol2_rh80=FILE_OUT, lookup_aerosol2_rh90=FILE_OUT, lookup_aerosol2_rh95=FILE_OUT,
      lookup_aerosol2_rh99=FILE_OUT)
def run_aerosol(mustCompile, mustClean, lookup_aerosol2_rh00,
                lookup_aerosol2_rh50, lookup_aerosol2_rh70,
                lookup_aerosol2_rh80, lookup_aerosol2_rh90,
                lookup_aerosol2_rh95, lookup_aerosol2_rh99):

    pass


"""
* ****************************************************************************
* ****************************************************************************
* ****************************************************************************
* ******************** VARIABLE STEP *****************************************
* ****************************************************************************
* ****************************************************************************
* ****************************************************************************
"""


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + BinaryWrapper.DEGRIB_GFS_GENERIC)
@task(returns=int, cW=FILE_OUT, iCEC=FILE_OUT, sH=FILE_OUT, sOILT2=FILE_OUT, sOILT4=FILE_OUT,
      sOILW2=FILE_OUT, sOILW4=FILE_OUT, tT=FILE_OUT, vV=FILE_OUT, hH=FILE_OUT,
      pRMSL=FILE_OUT, sOILT1=FILE_OUT, sOILT3=FILE_OUT, sOILW1=FILE_OUT,
      sOILW3=FILE_OUT, sST_TS=FILE_OUT, uU=FILE_OUT, wEASD=FILE_OUT)
def degribgfs_generic_05(cW, iCEC, sH, sOILT2, sOILT4, sOILW2, sOILW4, tT,
                         vV, hH, pRMSL, sOILT1, sOILT3, sOILW1, sOILW3,
                         sST_TS, uU, wEASD):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.GFS2MODEL + FortranWrapper.SUFFIX_EXE)
@task(returns=int, cW=FILE_IN, iCEC=FILE_IN, sH=FILE_IN, sOILT2=FILE_IN, sOILT4=FILE_IN,
      sOILW2=FILE_IN, sOILW4=FILE_IN, tT=FILE_IN, vV=FILE_IN,
      hH=FILE_IN, pRMSL=FILE_IN, sOILT1=FILE_IN, sOILT3=FILE_IN,
      sOILW1=FILE_IN, sOILW3=FILE_IN, sST_TS=FILE_IN, uU=FILE_IN,
      wEASD=FILE_IN, gFS_file=FILE_OUT)
def gfs2model_rrtm(cW, iCEC, sH, sOILT2, sOILT4, sOILW2, sOILW4, tT, vV,
                   hH, pRMSL, sOILT1, sOILT3, sOILW1, sOILW3, sST_TS, uU,
                   wEASD, gFS_file):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.INC_RRTM + FortranWrapper.SUFFIX_EXE)
@task(returns=int, gFS_file=FILE_IN)
def inc_rrtm(gFS_file, deco):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.CNV_RRTM + FortranWrapper.SUFFIX_EXE)
@task(returns=int, gFS_file=FILE_IN,
      llspl000=FILE_OUT, outtmp=FILE_OUT, outmst=FILE_OUT, outsst=FILE_OUT,
      outsno=FILE_OUT, outcic=FILE_OUT)
def cnv_rrtm(gFS_file, llspl000, outtmp, outmst, outsst, outsno, outcic):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.DEGRIB_SST + FortranWrapper.SUFFIX_EXE)
@task(returns=int, llgsst05=FILE_OUT)
def degribsst(llgsst05, sstfileinPath):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.ALBEDO + FortranWrapper.SUFFIX_EXE)
@task(returns=int, llspl000=FILE_IN, seamask=FILE_IN, Albedo=FILE_OUT, albedobase=FILE_OUT)
def albedo(llspl000, seamask, Albedo, albedobase, albedomnth):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.ALBEDO_RRTM_1DEG + FortranWrapper.SUFFIX_EXE)
@task(returns=int, llspl000=FILE_IN, seamask=FILE_IN, albedo_rrtm=FILE_OUT)
def albedorrtm(llspl000, seamask, albedo_rrtm, albedorrtm1degDir):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.VEG_FRAC + FortranWrapper.SUFFIX_EXE)
@task(returns=int, llspl000=FILE_IN, seamask=FILE_IN, veg_frac=FILE_OUT)
def vegfrac(llspl000, seamask, veg_frac, vegfracmnth):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.Z0_VEGUSTAR + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamask=FILE_IN, land_use=FILE_IN, top_soil_type=FILE_IN, height=FILE_IN,
      st_dh=FILE_IN, veg_frac=FILE_IN,
      z0base=FILE_OUT, z0=FILE_OUT, ustar=FILE_OUT)
def z0vegfrac(seamask, land_use, top_soil_type, height, st_dh, veg_frac, z0base,
              z0, ustar):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.ALLPREP_RRTM + FortranWrapper.SUFFIX_EXE)
@task(returns=int,
      llspl000=FILE_IN,
      llgsst05=FILE_IN,
      sst05=FILE_OUT,
      height=FILE_IN,
      seamask=FILE_IN,
      st_dh=FILE_IN,
      deep_temperature=FILE_INOUT,
      snow_albedo=FILE_INOUT,
      z0=FILE_IN,
      z0base=FILE_IN,
      land_use=FILE_IN,
      land_use_new=FILE_IN,
      top_soil_type=FILE_IN,
      veg_frac=FILE_IN,
      albedo_rrtm=FILE_IN,
      llgsst=FILE_IN,
      lgsno=FILE_IN,
      llgcic=FILE_IN,
      llsmst=FILE_IN,
      llstmp=FILE_IN,
      albedorrtmcorr=FILE_OUT,
      dzsoil=FILE_OUT,
      tskin=FILE_OUT,
      sst=FILE_OUT,
      snow=FILE_OUT,
      snowheight=FILE_OUT,
      cice=FILE_OUT,
      seamaskcorr=FILE_OUT,
      landusecorr=FILE_OUT,
      landusenewcorr=FILE_OUT,
      top_soil_typecorr=FILE_OUT,
      vegfraccorr=FILE_OUT,
      z0corr=FILE_OUT,
      z0basecorr=FILE_OUT,
      emissivity=FILE_OUT,
      canopywater=FILE_OUT,
      frozenprecratio=FILE_OUT,
      smst=FILE_OUT,
      sh2o=FILE_OUT,
      stmp=FILE_OUT,
      dsg=FILE_IN,
      fcst=FILE_OUT,
      Albedo=FILE_IN,
      ustar=FILE_IN)
def allprep(llspl000, llgsst05, sst05, height, seamask, st_dh, deep_temperature,
            snow_albedo, z0, z0base, land_use, land_use_new, top_soil_type,
            veg_frac, albedo_rrtm, llgsst, lgsno, llgcic, llsmst, llstmp,
            albedorrtmcorr, dzsoil, tskin, sst, snow, snowheight, cice,
            seamaskcorr, landusecorr, landusenewcorr, top_soil_typecorr,
            vegfraccorr, z0corr, z0basecorr, emissivity, canopywater,
            frozenprecratio, smst, sh2o, stmp, dsg, fcst, Albedo, ustar,
            fcstDir, bocoPrefix, llsplPrefix):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.READ_PAUL_SOURCE + FortranWrapper.SUFFIX_EXE)
@task(returns=int, seamask=FILE_IN, source=FILE_OUT, sourceNETCDF=FILE_OUT)
def readpaulsource(seamask, source, sourceNETCDF, sourceNCIncludeDir):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.DUST_START + FortranWrapper.SUFFIX_EXE)
@task(returns=int,
      llspl000=FILE_IN,
      soildust=FILE_OUT,
      snow=FILE_IN,
      top_soil_typecorr=FILE_IN,
      landusecorr=FILE_IN,
      landusenewcorr=FILE_IN,
      kount_land_use=FILE_IN,
      kount_land_use_new=FILE_IN,
      veg_frac=FILE_IN,
      height=FILE_IN,
      seamask=FILE_IN,
      source=FILE_IN,
      z0corr=FILE_IN,
      Roughness=FILE_IN)
def dust_start(llspl000, soildust, snow, top_soil_typecorr, landusecorr,
               landusenewcorr, kount_land_use, kount_land_use_new, veg_frac,
               height, seamask, source, z0corr, Roughness):
    pass


"""
* ****************************************************************************
* ****************************************************************************
* ****************************************************************************
* ******************** POST PROC STEP ****************************************
* ****************************************************************************
* ****************************************************************************
* ****************************************************************************
"""


@binary(binary="${" + NMMBConstants.ENV_NAME_POST_CARBONO + "}/" + BinaryWrapper.PREPARE_POSTPROC_AUTH)
@task(returns=int)
def preparePost(mustCompile, folderOutput):
    pass


@binary(binary="${" + NMMBConstants.ENV_NAME_POST_CARBONO + "}/" + BinaryWrapper.EXEC_POSTPROC_AUTH)
@task(returns=int, destPath=FILE_OUT)
def executePostprocAuth(folderOutput, sourcesPath, destPath):
    pass
