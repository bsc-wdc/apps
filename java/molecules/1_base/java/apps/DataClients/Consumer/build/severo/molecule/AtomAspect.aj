package severo.molecule;
public aspect AtomAspect {
    pointcut pointSetPC(): !withincode(void severo.molecule.Atom.$$setpoint(..))  && !withincode(void severo.molecule.Atom.serialize*(..)) && !withincode(void severo.molecule.Atom.deserialize*(..)) && set(*  severo.molecule.Atom.point);
    void around(severo.molecule.Atom instance, severo.molecule.Point newval): pointSetPC() && args(newval)  && target(instance) { instance.$$setpoint(newval);
}
    pointcut pointGetPC(): !withincode(* severo.molecule.Atom.$$getpoint())  && !withincode(void severo.molecule.Atom.serialize*(..)) && !withincode(void severo.molecule.Atom.deserialize*(..)) && get(*  severo.molecule.Atom.point);
    severo.molecule.Point around(severo.molecule.Atom instance): pointGetPC()  && target(instance) { severo.molecule.Point val = instance.$$getpoint(); return val; }
    pointcut elementNameSetPC(): !withincode(void severo.molecule.Atom.$$setelementName(..))  && !withincode(void severo.molecule.Atom.serialize*(..)) && !withincode(void severo.molecule.Atom.deserialize*(..)) && set(*  severo.molecule.Atom.elementName);
    void around(severo.molecule.Atom instance, java.lang.String newval): elementNameSetPC() && args(newval)  && target(instance) { instance.$$setelementName(newval);
}
    pointcut elementNameGetPC(): !withincode(* severo.molecule.Atom.$$getelementName())  && !withincode(void severo.molecule.Atom.serialize*(..)) && !withincode(void severo.molecule.Atom.deserialize*(..)) && get(*  severo.molecule.Atom.elementName);
    java.lang.String around(severo.molecule.Atom instance): elementNameGetPC()  && target(instance) { java.lang.String val = instance.$$getelementName(); return val; }
    pointcut massSetPC(): !withincode(void severo.molecule.Atom.$$setmass(..))  && !withincode(void severo.molecule.Atom.serialize*(..)) && !withincode(void severo.molecule.Atom.deserialize*(..)) && set(*  severo.molecule.Atom.mass);
    void around(severo.molecule.Atom instance, float newval): massSetPC() && args(newval)  && target(instance) { instance.$$setmass(newval);
}
    pointcut massGetPC(): !withincode(* severo.molecule.Atom.$$getmass())  && !withincode(void severo.molecule.Atom.serialize*(..)) && !withincode(void severo.molecule.Atom.deserialize*(..)) && get(*  severo.molecule.Atom.mass);
    float around(severo.molecule.Atom instance): massGetPC()  && target(instance) { float val = instance.$$getmass(); return val; }
}
