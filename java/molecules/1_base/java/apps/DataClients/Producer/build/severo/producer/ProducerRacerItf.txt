
package severo.producer;

import es.bsc.compss.types.annotations.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import severo.molecule.Molecule;

public interface ProducerRacerItf {

	@Method(declaringClass = "severo.producer.ProducerRacer")
	boolean checkExitCondition(
			@Parameter(type = Type.OBJECT, direction = Direction.IN)
			Molecule molecule,
			@Parameter(type = Type.INT, direction = Direction.IN)
			int m);			
}