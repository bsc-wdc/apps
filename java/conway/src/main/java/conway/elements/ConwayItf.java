package conway.elements;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;


public interface ConwayItf {

    @Constraints(computingUnits = "1", memorySize = "0.5")
    @Method(declaringClass = "conway.elements.ConwayImpl")
    int updateCell(
        @Parameter(type = Type.OBJECT, direction = Direction.IN) State state_A,
        @Parameter(type = Type.INT, direction = Direction.IN) int i,
        @Parameter(type = Type.INT, direction = Direction.IN) int j
    );
}