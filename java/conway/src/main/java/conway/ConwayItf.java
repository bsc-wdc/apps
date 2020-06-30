package conway;

import  es.bsc.compss.types.annotations.Constraints;
import  es.bsc.compss.types.annotations.task.Method;
import  es.bsc.compss.types.annotations.Parameter;
import  es.bsc.compss.types.annotations.parameter.Direction;
import  es.bsc.compss.types.annotations.parameter.Type;

public interface ConwayItf {
 // @Constraints(computingUnits = "1", memorySize = "0.5")
  @Method(declaringClass = "conway.ConwayImpl")
  void update_block (
      @Parameter(type = Type.OBJECT, direction = Direction.IN) Zone Z,
      @Parameter(type = Type.OBJECT, direction = Direction.INOUT) Block T
  );
}