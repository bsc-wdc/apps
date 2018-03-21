package severo.producer;

import es.bsc.compss.types.annotations.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.Parameter.Type;

import severo.moleculeArray.Molecule;

public interface ProducerDataClayItf {

    @Method(declaringClass = "severo.moleculeArray.Molecule")
    void computeCenterOfMass();

    @Method(declaringClass = "severo.moleculeArray.Molecule")
    void init(
            @Parameter int n
    );
    
    @Method(declaringClass = "severo.moleculeArray.Molecule")
    void makePersistent(
            @Parameter(type = Type.STRING) String alias
    );
    
} 
