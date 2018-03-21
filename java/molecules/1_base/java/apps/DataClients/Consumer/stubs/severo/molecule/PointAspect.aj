package severo.molecule;
public aspect PointAspect {
    pointcut xSetPC(): !withincode(void severo.molecule.Point.$$setx(..))  && !withincode(void severo.molecule.Point.serialize*(..)) && !withincode(void severo.molecule.Point.deserialize*(..)) && set(*  severo.molecule.Point.x);
    void around(severo.molecule.Point instance, float newval): xSetPC() && args(newval)  && target(instance) { instance.$$setx(newval);
}
    pointcut xGetPC(): !withincode(* severo.molecule.Point.$$getx())  && !withincode(void severo.molecule.Point.serialize*(..)) && !withincode(void severo.molecule.Point.deserialize*(..)) && get(*  severo.molecule.Point.x);
    float around(severo.molecule.Point instance): xGetPC()  && target(instance) { float val = instance.$$getx(); return val; }
    pointcut ySetPC(): !withincode(void severo.molecule.Point.$$sety(..))  && !withincode(void severo.molecule.Point.serialize*(..)) && !withincode(void severo.molecule.Point.deserialize*(..)) && set(*  severo.molecule.Point.y);
    void around(severo.molecule.Point instance, float newval): ySetPC() && args(newval)  && target(instance) { instance.$$sety(newval);
}
    pointcut yGetPC(): !withincode(* severo.molecule.Point.$$gety())  && !withincode(void severo.molecule.Point.serialize*(..)) && !withincode(void severo.molecule.Point.deserialize*(..)) && get(*  severo.molecule.Point.y);
    float around(severo.molecule.Point instance): yGetPC()  && target(instance) { float val = instance.$$gety(); return val; }
    pointcut zSetPC(): !withincode(void severo.molecule.Point.$$setz(..))  && !withincode(void severo.molecule.Point.serialize*(..)) && !withincode(void severo.molecule.Point.deserialize*(..)) && set(*  severo.molecule.Point.z);
    void around(severo.molecule.Point instance, float newval): zSetPC() && args(newval)  && target(instance) { instance.$$setz(newval);
}
    pointcut zGetPC(): !withincode(* severo.molecule.Point.$$getz())  && !withincode(void severo.molecule.Point.serialize*(..)) && !withincode(void severo.molecule.Point.deserialize*(..)) && get(*  severo.molecule.Point.z);
    float around(severo.molecule.Point instance): zGetPC()  && target(instance) { float val = instance.$$getz(); return val; }
}
