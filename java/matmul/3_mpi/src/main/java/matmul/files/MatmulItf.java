package matmul.files;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.MPI;
import es.bsc.compss.types.annotations.task.Method;


public interface MatmulItf {
    
    @Method(declaringClass = "matmul.files.MatmulImpl")
    @Constraints(computingUnits = "1")
    void initializeBlock(
        @Parameter(type = Type.FILE, direction = Direction.OUT) String filename, 
        @Parameter() int BSIZE, 
        @Parameter() boolean initRand
    );

    @Method(declaringClass = "matmul.files.MatmulImpl")
    @Constraints(computingUnits = "${CUS}")
    Integer multiplyAccumulativeNative(
        @Parameter() int bsize, 
        @Parameter(type = Type.FILE, direction = Direction.IN) String aIn,
        @Parameter(type = Type.FILE, direction = Direction.IN) String bIn,
        @Parameter(type = Type.FILE, direction = Direction.INOUT) String cOut
    );
    
    @MPI(binary = "${MATMUL_BINARY}", 
         mpiRunner = "mpirun", 
         processes = "${MPI_PROCS}", scaleByCU = true)
    @Constraints(computingUnits = "${CUS}")
    Integer multiplyAccumulativeMPI(
       @Parameter() int bsize, 
       @Parameter(type = Type.FILE, direction = Direction.IN) String aIn,
       @Parameter(type = Type.FILE, direction = Direction.IN) String bIn,
       @Parameter(type = Type.FILE, direction = Direction.INOUT) String cOut
    );

}
