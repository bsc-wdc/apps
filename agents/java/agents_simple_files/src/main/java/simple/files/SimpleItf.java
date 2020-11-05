package simple.files;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.Method;


public interface SimpleItf {

    @Method(declaringClass = "simple.files.Simple")
    void increment(
        @Parameter(direction = Direction.INOUT, type = Type.FILE) String filePath
    );

}
