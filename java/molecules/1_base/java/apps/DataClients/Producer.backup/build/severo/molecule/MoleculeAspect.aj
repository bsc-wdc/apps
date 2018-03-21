package severo.molecule;
public aspect MoleculeAspect {
    pointcut nameSetPC(): !withincode(void severo.molecule.Molecule.$$setname(..))  && !withincode(void severo.molecule.Molecule.serialize*(..)) && !withincode(void severo.molecule.Molecule.deserialize*(..)) && set(*  severo.molecule.Molecule.name);
    void around(severo.molecule.Molecule instance, java.lang.String newval): nameSetPC() && args(newval)  && target(instance) { instance.$$setname(newval);
}
    pointcut nameGetPC(): !withincode(* severo.molecule.Molecule.$$getname())  && !withincode(void severo.molecule.Molecule.serialize*(..)) && !withincode(void severo.molecule.Molecule.deserialize*(..)) && get(*  severo.molecule.Molecule.name);
    java.lang.String around(severo.molecule.Molecule instance): nameGetPC()  && target(instance) { java.lang.String val = instance.$$getname(); return val; }
    pointcut atomsSetPC(): !withincode(void severo.molecule.Molecule.$$setatoms(..))  && !withincode(void severo.molecule.Molecule.serialize*(..)) && !withincode(void severo.molecule.Molecule.deserialize*(..)) && set(*  severo.molecule.Molecule.atoms);
    void around(severo.molecule.Molecule instance, severo.molecule.Atom[] newval): atomsSetPC() && args(newval)  && target(instance) { instance.$$setatoms(newval);
}
    pointcut atomsGetPC(): !withincode(* severo.molecule.Molecule.$$getatoms())  && !withincode(void severo.molecule.Molecule.serialize*(..)) && !withincode(void severo.molecule.Molecule.deserialize*(..)) && get(*  severo.molecule.Molecule.atoms);
    severo.molecule.Atom[] around(severo.molecule.Molecule instance): atomsGetPC()  && target(instance) { severo.molecule.Atom[] val = instance.$$getatoms(); return val; }
    pointcut centerSetPC(): !withincode(void severo.molecule.Molecule.$$setcenter(..))  && !withincode(void severo.molecule.Molecule.serialize*(..)) && !withincode(void severo.molecule.Molecule.deserialize*(..)) && set(*  severo.molecule.Molecule.center);
    void around(severo.molecule.Molecule instance, severo.molecule.Atom newval): centerSetPC() && args(newval)  && target(instance) { instance.$$setcenter(newval);
}
    pointcut centerGetPC(): !withincode(* severo.molecule.Molecule.$$getcenter())  && !withincode(void severo.molecule.Molecule.serialize*(..)) && !withincode(void severo.molecule.Molecule.deserialize*(..)) && get(*  severo.molecule.Molecule.center);
    severo.molecule.Atom around(severo.molecule.Molecule instance): centerGetPC()  && target(instance) { severo.molecule.Atom val = instance.$$getcenter(); return val; }
}
