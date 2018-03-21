package blast;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.Binary;
import es.bsc.compss.types.annotations.task.Method;


public interface BlastItf {
    
    @Method(declaringClass = "blast.BlastImpl")
    void splitPartitions(
        @Parameter(type = Type.FILE, direction = Direction.IN) String inputFileName, 
        @Parameter(type = Type.FILE, direction = Direction.OUT) String partitionFile, 
        @Parameter() int nFrags, 
        @Parameter() int myFrag
    );
    
    @Binary(binary = "${BLAST_BINARY}")
    Integer align(
        @Parameter(type = Type.STRING, direction = Direction.IN) String pFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String pMode,
        @Parameter(type = Type.STRING, direction = Direction.IN) String dFlag,
        @Parameter(type = Type.STRING, direction = Direction.IN) String database,
        @Parameter(type = Type.STRING, direction = Direction.IN) String iFlag,
        @Parameter(type = Type.FILE, direction = Direction.IN) String partitionFile,
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String partitionOutput,
        @Parameter(type = Type.STRING, direction = Direction.IN) String extraCMDArgs
    );
    
    @Binary(binary = "${BLAST_BINARY}")
    Integer align(
        @Parameter(type = Type.STRING, direction = Direction.IN) String pFlag, 
        @Parameter(type = Type.STRING, direction = Direction.IN) String pMode,
        @Parameter(type = Type.STRING, direction = Direction.IN) String dFlag,
        @Parameter(type = Type.STRING, direction = Direction.IN) String database,
        @Parameter(type = Type.STRING, direction = Direction.IN) String iFlag,
        @Parameter(type = Type.FILE, direction = Direction.IN) String partitionFile,
        @Parameter(type = Type.STRING, direction = Direction.IN) String oFlag,
        @Parameter(type = Type.FILE, direction = Direction.OUT) String partitionOutput
    );
    
	@Method(declaringClass = "blast.BlastImpl")
	void assemblyPartitions(
		@Parameter(type = Type.FILE, direction = Direction.INOUT) String partialFileA,
		@Parameter(type = Type.FILE, direction = Direction.IN) String partialFileB
	);
	
}
