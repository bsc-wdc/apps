package severo.consumer;

import es.bsc.compss.types.annotations.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.Parameter.Direction;
import es.bsc.compss.types.annotations.Parameter.Type;
import severo.molecule.Molecule;

public interface ConsumerRacerItf {

	@Method(declaringClass = "severo.consumer.ConsumerRacer")
	void doMoleculeOffsetX(
			@Parameter(type = Type.OBJECT, direction = Direction.INOUT)
			Molecule molecule,
			@Parameter(type = Type.INT, direction = Direction.IN)
			int m);

	@Method(declaringClass = "severo.consumer.ConsumerRacer")
	void doMoleculeOffsetY(
			@Parameter(type = Type.OBJECT, direction = Direction.INOUT)
			Molecule molecule,
			@Parameter(type = Type.INT, direction = Direction.IN)
			int m);

	@Method(declaringClass = "severo.consumer.ConsumerRacer")
	void doMoleculeOffsetZ(
			@Parameter(type = Type.OBJECT, direction = Direction.INOUT)
			Molecule molecule,
			@Parameter(type = Type.INT, direction = Direction.IN)
			int m);

	@Method(declaringClass = "severo.molecule.Molecule")
	void computeCenterOfMass();;

}
