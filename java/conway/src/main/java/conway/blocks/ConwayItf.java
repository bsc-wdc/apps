package conway.blocks;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;

public interface ConwayItf {
    
    @Method(declaringClass = "conway.blocks.ConwayImpl")
    Block initBlock(
        @Parameter() int blockSize
    );
    
    @Method(declaringClass = "conway.blocks.ConwayImpl")
    Block updateBlock(
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b00, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b01, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b02, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b10, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b11, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b12, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b20,
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b21, 
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Block b22, 
        @Parameter() int aFactor, 
        @Parameter() int bSize
    );

}