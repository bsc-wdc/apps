package severo.moleculeArray;
public aspect MoleculeAspect {
    pointcut nameSetPC(): !withincode(void severo.moleculeArray.Molecule.$$setname(..))  && !withincode(void severo.moleculeArray.Molecule.serialize*(..)) && !withincode(void severo.moleculeArray.Molecule.deserialize*(..)) && set(*  severo.moleculeArray.Molecule.name);
    void around(severo.moleculeArray.Molecule instance, java.lang.String newval): nameSetPC() && args(newval)  && target(instance) { instance.$$setname(newval);
}
    pointcut nameGetPC(): !withincode(* severo.moleculeArray.Molecule.$$getname())  && !withincode(void severo.moleculeArray.Molecule.serialize*(..)) && !withincode(void severo.moleculeArray.Molecule.deserialize*(..)) && get(*  severo.moleculeArray.Molecule.name);
    java.lang.String around(severo.moleculeArray.Molecule instance): nameGetPC()  && target(instance) { java.lang.String val = instance.$$getname(); return val; }
    pointcut atomsSetPC(): !withincode(void severo.moleculeArray.Molecule.$$setatoms(..))  && !withincode(void severo.moleculeArray.Molecule.serialize*(..)) && !withincode(void severo.moleculeArray.Molecule.deserialize*(..)) && set(*  severo.moleculeArray.Molecule.atoms);
    void around(severo.moleculeArray.Molecule instance, float[][] newval): atomsSetPC() && args(newval)  && target(instance) { instance.$$setatoms(newval);
}
    pointcut atomsGetPC(): !withincode(* severo.moleculeArray.Molecule.$$getatoms())  && !withincode(void severo.moleculeArray.Molecule.serialize*(..)) && !withincode(void severo.moleculeArray.Molecule.deserialize*(..)) && get(*  severo.moleculeArray.Molecule.atoms);
    float[][] around(severo.moleculeArray.Molecule instance): atomsGetPC()  && target(instance) { float[][] val = instance.$$getatoms(); return val; }
    pointcut centerSetPC(): !withincode(void severo.moleculeArray.Molecule.$$setcenter(..))  && !withincode(void severo.moleculeArray.Molecule.serialize*(..)) && !withincode(void severo.moleculeArray.Molecule.deserialize*(..)) && set(*  severo.moleculeArray.Molecule.center);
    void around(severo.moleculeArray.Molecule instance, float[] newval): centerSetPC() && args(newval)  && target(instance) { instance.$$setcenter(newval);
}
    pointcut centerGetPC(): !withincode(* severo.moleculeArray.Molecule.$$getcenter())  && !withincode(void severo.moleculeArray.Molecule.serialize*(..)) && !withincode(void severo.moleculeArray.Molecule.deserialize*(..)) && get(*  severo.moleculeArray.Molecule.center);
    float[] around(severo.moleculeArray.Molecule instance): centerGetPC()  && target(instance) { float[] val = instance.$$getcenter(); return val; }
}
