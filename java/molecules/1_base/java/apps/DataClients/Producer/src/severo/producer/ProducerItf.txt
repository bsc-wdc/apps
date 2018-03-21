
package severo.producer;

import es.bsc.compss.types.annotations.Method;


public interface ProducerItf {
	
	@Method(declaringClass = "severo.molecule.Molecule")
	void computeCenterOfMass();

}
