package conway.blocks;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;


public interface ConwayItf {

    // @Constraints(computingUnits = "1", memorySize = "0.5")
    @Method(declaringClass = "conway.blocks.ConwayImpl")
    Block updateBlock(
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Zone z
    );
    
}