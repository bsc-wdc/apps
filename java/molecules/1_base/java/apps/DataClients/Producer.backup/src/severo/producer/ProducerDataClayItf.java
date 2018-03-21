package severo.producer;

import es.bsc.compss.types.annotations.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.Parameter.Type;

import severo.molecule.Molecule;

public interface ProducerDataClayItf {

    @Method(declaringClass = "severo.molecule.Molecule")
    void computeCenterOfMass();

    @Method(declaringClass = "severo.molecule.Molecule")
    void init(
            @Parameter int n
    );
    
    @Method(declaringClass = "severo.molecule.Molecule")
    void makePersistent(
            @Parameter(type = Type.STRING) String alias
    );
    
}

