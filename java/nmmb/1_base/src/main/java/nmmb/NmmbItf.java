package nmmb;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Stream;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.Binary;
import es.bsc.compss.types.annotations.task.MPI;
import nmmb.configuration.NMMBConstants;
import nmmb.utils.BinaryWrapper;
import nmmb.utils.FortranWrapper;


public interface NmmbItf {
    
    /*
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ******************** COMPILE STEP *****************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     */
    @Binary(binary = FortranWrapper.FC)
    Integer fortranCompiler(
        @Parameter(type = Type.STRING, direction = Direction.IN) String mcFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String sharedFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String covertPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String convertValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String tracebackFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumePrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumeValue, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String optFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String stackFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String executable, 
        @Parameter(type = Type.FILE, direction = Direction.IN) String source
    );
    
    @Binary(binary = FortranWrapper.FC)
    Integer fortranCompileObject(
        @Parameter(type = Type.STRING, direction = Direction.IN) String mcFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String sharedFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String covertPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String convertValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String tracebackFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumePrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumeValue, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String optFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String stackFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String cFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String source, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String object,
        @Parameter(type = Type.STRING, direction = Direction.IN) String moduleFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String moduleDir
    );
    
    @Binary(binary = FortranWrapper.FC)
    Integer fortranCompileWithObject(
        @Parameter(type = Type.STRING, direction = Direction.IN) String mcFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String sharedFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String covertPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String convertValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String tracebackFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumePrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumeValue, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String optFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String stackFlag,
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String executable,
        @Parameter(type = Type.STRING, direction = Direction.IN) String source, 
        @Parameter(type = Type.FILE, direction = Direction.IN) String object
    );
    
    @Binary(binary = FortranWrapper.FC)
    Integer fortranCompilerWithW3(
        @Parameter(type = Type.STRING, direction = Direction.IN) String mcFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String sharedFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String covertPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String convertValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String tracebackFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumePrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String assumeValue, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String optFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String fpmodelValue,
        @Parameter(type = Type.STRING, direction = Direction.IN) String stackFlag,
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String executable, 
        @Parameter(type = Type.FILE, direction = Direction.IN) String source,
        @Parameter(type = Type.STRING, direction = Direction.IN) String w3libDir, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String bacioLibDir,
        @Parameter(type = Type.STRING, direction = Direction.IN) String w3Lib,
        @Parameter(type = Type.STRING, direction = Direction.IN) String bacioLib
    );
    
    @Binary(binary = FortranWrapper.GFC)
    Integer gfortranCompiler(
        @Parameter(type = Type.STRING, direction = Direction.IN) String bigOFlag, 
        @Parameter(type = Type.FILE, direction = Direction.IN) String source, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String executable
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + BinaryWrapper.COMPILE_READ_PAUL_SOURCE)
    Integer compileReadPaulSource(
        @Parameter(type = Type.FILE, direction = Direction.IN) String source, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String executable
    );
    
    
    /*
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ******************** FIXED STEP *******************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     */
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.SMMOUNT + FortranWrapper.SUFFIX_EXE)
    Integer smmount(
        @Parameter(type = Type.STRING, direction = Direction.IN) String topoDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String seamaskDEM,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String heightDEM
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.LANDUSE + FortranWrapper.SUFFIX_EXE)
    Integer landuse(
        @Parameter(type = Type.STRING, direction = Direction.IN) String landuseDir,  
        @Parameter(type = Type.FILE, direction = Direction.OUT) String landuse,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String kount_landuse
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.LANDUSENEW + FortranWrapper.SUFFIX_EXE)
    Integer landusenew(
        @Parameter(type = Type.STRING, direction = Direction.IN) String gtopDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String landusenew,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String kount_landusenew
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.TOPO + FortranWrapper.SUFFIX_EXE)
    Integer topo(
        @Parameter(type = Type.STRING, direction = Direction.IN) String topoDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String heightmean
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.STDH + FortranWrapper.SUFFIX_EXE)
    Integer stdh(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamaskDEM,
        @Parameter(type = Type.FILE, direction = Direction.IN) String heightmean,
        @Parameter(type = Type.STRING, direction = Direction.IN) String topoDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String stdh
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.ENVELOPE + FortranWrapper.SUFFIX_EXE)
    Integer envelope(
        @Parameter(type = Type.FILE, direction = Direction.IN) String heightmean,
        @Parameter(type = Type.FILE, direction = Direction.IN) String stdh,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String height
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.TOPSOILTYPE + FortranWrapper.SUFFIX_EXE)
    Integer topsoiltype(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamaskDEM,
        @Parameter(type = Type.STRING, direction = Direction.IN) String soiltypeDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String topsoiltype
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.BOTSOILTYPE + FortranWrapper.SUFFIX_EXE)
    Integer botsoiltype(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamaskDEM,
        @Parameter(type = Type.STRING, direction = Direction.IN) String soiltypePath,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String botsoiltype
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.TOPOSEAMASK + FortranWrapper.SUFFIX_EXE)
    Integer toposeamask(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamaskDEM,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String height,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String landuse,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String topsoiltype,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String botsoiltype
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.STDHTOPO + FortranWrapper.SUFFIX_EXE)
    Integer stdhtopo(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String stdh
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.DEEPTEMPERATURE + FortranWrapper.SUFFIX_EXE)
    Integer deeptemperature(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.STRING, direction = Direction.IN) String soiltempPath, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String deeptemperature
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.SNOWALBEDO + FortranWrapper.SUFFIX_EXE)
    Integer snowalbedo(
        @Parameter(type = Type.STRING, direction = Direction.IN) String maxsnowalbDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String snowalbedo
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.VCGENERATOR + FortranWrapper.SUFFIX_EXE)
    Integer vcgenerator(
        @Parameter(type = Type.FILE, direction = Direction.OUT) String dsg
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.ROUGHNESS + FortranWrapper.SUFFIX_EXE)
    Integer roughness(
        @Parameter(type = Type.STRING, direction = Direction.IN) String roughnessDir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String roughness
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/" + FortranWrapper.GFDLCO2 + FortranWrapper.SUFFIX_EXE)
    Integer gfdlco2(
        @Parameter(type = Type.FILE, direction = Direction.IN) String dsg,
        @Parameter(type = Type.STRING, direction = Direction.IN) String co2_data_dir,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String co2_trans
    );

    @Binary(binary = "${" + NMMBConstants.ENV_NAME_FIX + "}/lookup_tables/" + BinaryWrapper.RUN_AEROSOL)
    Integer run_aerosol(
        @Parameter() boolean mustCompile, 
        @Parameter() boolean mustClean, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh00,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh50, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh70, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh80,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh90, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh95, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String lookup_aerosol2_rh99
    );
    
    
    /*
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ******************** VARIABLE STEP ****************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     */
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + BinaryWrapper.DEGRIB_GFS_GENERIC)
    Integer degribgfs_generic_05(
        @Parameter(type = Type.FILE, direction = Direction.OUT) String CW,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String ICEC,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SH,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILT2,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILT4,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILW2,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILW4,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String TT,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String VV,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String HH,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String PRMSL,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILT1,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILT3,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILW1,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SOILW3,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String SST_TS,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String UU,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String WEASD
    );    
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.GFS2MODEL + FortranWrapper.SUFFIX_EXE)
    Integer gfs2model_rrtm(
        @Parameter(type = Type.FILE, direction = Direction.IN) String CW,
        @Parameter(type = Type.FILE, direction = Direction.IN) String ICEC,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SH,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILT2,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILT4,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILW2,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILW4,
        @Parameter(type = Type.FILE, direction = Direction.IN) String TT,
        @Parameter(type = Type.FILE, direction = Direction.IN) String VV,
        @Parameter(type = Type.FILE, direction = Direction.IN) String HH,
        @Parameter(type = Type.FILE, direction = Direction.IN) String PRMSL,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILT1,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILT3,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILW1,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SOILW3,
        @Parameter(type = Type.FILE, direction = Direction.IN) String SST_TS,
        @Parameter(type = Type.FILE, direction = Direction.IN) String UU,
        @Parameter(type = Type.FILE, direction = Direction.IN) String WEASD,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String GFS_file
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.INC_RRTM + FortranWrapper.SUFFIX_EXE)
    Integer inc_rrtm(
        @Parameter(type = Type.FILE, direction = Direction.IN) String GFS_file,
        @Parameter(type = Type.STRING, direction = Direction.IN) String deco
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.CNV_RRTM + FortranWrapper.SUFFIX_EXE)
    Integer cnv_rrtm(
        @Parameter(type = Type.FILE, direction = Direction.IN) String GFS_file,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String llspl000,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String outtmp,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String outmst,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String outsst,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String outsno,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String outcic
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.DEGRIB_SST + FortranWrapper.SUFFIX_EXE)
    Integer degribsst(
        @Parameter(type = Type.FILE, direction = Direction.OUT) String llgsst05,
        @Parameter(type = Type.STRING, direction = Direction.IN) String sstfileinPath
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.ALBEDO + FortranWrapper.SUFFIX_EXE)
    Integer albedo(
        @Parameter(type = Type.FILE, direction = Direction.IN) String llspl000,
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String albedo,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String albedobase,
        @Parameter(type = Type.STRING, direction = Direction.IN) String albedomnth
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.ALBEDO_RRTM_1DEG + FortranWrapper.SUFFIX_EXE)
    Integer albedorrtm(
        @Parameter(type = Type.FILE, direction = Direction.IN) String llspl000,
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String albedorrtm,
        @Parameter(type = Type.STRING, direction = Direction.IN) String albedorrtm1degDir
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.VEG_FRAC + FortranWrapper.SUFFIX_EXE)
    Integer vegfrac(
        @Parameter(type = Type.FILE, direction = Direction.IN) String llspl000,
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String vegfrac,
        @Parameter(type = Type.STRING, direction = Direction.IN) String vegfracmnth
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.Z0_VEGUSTAR + FortranWrapper.SUFFIX_EXE)
    Integer z0vegfrac(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.IN) String landuse,
        @Parameter(type = Type.FILE, direction = Direction.IN) String topsoiltype,
        @Parameter(type = Type.FILE, direction = Direction.IN) String height,
        @Parameter(type = Type.FILE, direction = Direction.IN) String stdh,
        @Parameter(type = Type.FILE, direction = Direction.IN) String vegfrac,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String z0base,      
        @Parameter(type = Type.FILE, direction = Direction.OUT) String z0,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String ustar
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.ALLPREP_RRTM + FortranWrapper.SUFFIX_EXE)
    Integer allprep(
        @Parameter(type = Type.FILE, direction = Direction.IN) String llspl000,
        @Parameter(type = Type.FILE, direction = Direction.IN) String llgsst05,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String sst05,
        @Parameter(type = Type.FILE, direction = Direction.IN) String height,
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.IN) String stdh,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String deeptemperature,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String snowalbedo,
        @Parameter(type = Type.FILE, direction = Direction.IN) String z0,
        @Parameter(type = Type.FILE, direction = Direction.IN) String z0base,
        @Parameter(type = Type.FILE, direction = Direction.IN) String landuse,
        @Parameter(type = Type.FILE, direction = Direction.IN) String landusenew,
        @Parameter(type = Type.FILE, direction = Direction.IN) String topsoiltype,
        @Parameter(type = Type.FILE, direction = Direction.IN) String vegfrac,
        @Parameter(type = Type.FILE, direction = Direction.IN) String albedorrtm,
        @Parameter(type = Type.FILE, direction = Direction.IN) String llgsst,
        @Parameter(type = Type.FILE, direction = Direction.IN) String llgsno,
        @Parameter(type = Type.FILE, direction = Direction.IN) String llgcic,
        @Parameter(type = Type.FILE, direction = Direction.IN) String llsmst,
        @Parameter(type = Type.FILE, direction = Direction.IN) String llstmp,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String albedorrtmcorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String dzsoil,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String tskin,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String sst,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String snow,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String snowheight,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String cice,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String seamaskcorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String landusecorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String landusenewcorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String topsoiltypecorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String vegfraccorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String z0corr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String z0basecorr,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String emissivity,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String canopywater,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String frozenprecratio,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String smst,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String sh2o,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String stmp,
        @Parameter(type = Type.FILE, direction = Direction.IN) String dsg,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String fcst,
        @Parameter(type = Type.FILE, direction = Direction.IN) String albedo,
        @Parameter(type = Type.FILE, direction = Direction.IN) String ustar,
        @Parameter(type = Type.STRING, direction = Direction.IN) String fcstDir, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String bocoPrefix, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String llsplPrefix
    );
    
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.READ_PAUL_SOURCE + FortranWrapper.SUFFIX_EXE)
    Integer readpaulsource(
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String source,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String sourceNETCDF,
        @Parameter(type = Type.STRING, direction = Direction.IN) String sourceNCIncludeDir
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_VRB + "}/" + FortranWrapper.DUST_START + FortranWrapper.SUFFIX_EXE)
    Integer dust_start(
        @Parameter(type = Type.FILE, direction = Direction.IN) String llspl000,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String soildust,
        @Parameter(type = Type.FILE, direction = Direction.IN) String snow,
        @Parameter(type = Type.FILE, direction = Direction.IN) String topsoiltypecorr,
        @Parameter(type = Type.FILE, direction = Direction.IN) String landusecorr,
        @Parameter(type = Type.FILE, direction = Direction.IN) String landusenewcorr,
        @Parameter(type = Type.FILE, direction = Direction.IN) String kount_landuse,       
        @Parameter(type = Type.FILE, direction = Direction.IN) String kount_landusenew,
        @Parameter(type = Type.FILE, direction = Direction.IN) String vegfrac,
        @Parameter(type = Type.FILE, direction = Direction.IN) String height,      
        @Parameter(type = Type.FILE, direction = Direction.IN) String seamask,
        @Parameter(type = Type.FILE, direction = Direction.IN) String source,          
        @Parameter(type = Type.FILE, direction = Direction.IN) String z0corr,
        @Parameter(type = Type.FILE, direction = Direction.IN) String roughness
    );
    
    /*
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ******************** UMO MODEL STEP ***************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     */
    @MPI(mpiRunner = "mpirun", 
         binary = "${" + NMMBConstants.ENV_NAME_SRCDIR + "}/exe/" + FortranWrapper.NEMS + FortranWrapper.SUFFIX_EXE, 
         workingDir = "${" + NMMBConstants.ENV_NAME_UMO_OUT + "}/",
         computingNodes = "${NEMS_NODES}")
    @Constraints(computingUnits = "${NEMS_CUS_PER_NODE}")
    Integer nems(
        @Parameter(type = Type.FILE, direction = Direction.OUT, stream = Stream.STDOUT) String stdOutFile, 
        @Parameter(type = Type.FILE, direction = Direction.OUT, stream = Stream.STDERR) String stdErrFile
    );
    
    /*
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ******************** POST PROC STEP ***************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     * ***************************************************************************************************
     */    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_POST_CARBONO + "}/" + BinaryWrapper.PREPARE_POSTPROC_AUTH)
    Integer preparePost(
        @Parameter() boolean mustCompile, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String folderOutput
    );
    
    @Binary(binary = "${" + NMMBConstants.ENV_NAME_POST_CARBONO + "}/" + BinaryWrapper.EXEC_POSTPROC_AUTH)
    Integer executePostprocAuth(
        @Parameter(type = Type.STRING, direction = Direction.IN) String folderOutput,
        @Parameter(type = Type.STRING, direction = Direction.IN) String sourcesPath, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String destFile
    );
    
}
