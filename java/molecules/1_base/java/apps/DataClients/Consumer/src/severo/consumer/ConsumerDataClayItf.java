
package severo.consumer;

import es.bsc.compss.types.annotations.Method;


public interface ConsumerDataClayItf {

	@Method(declaringClass = "severo.molecule.Molecule")
	void printCenterOfMass();
	
}
