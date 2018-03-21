package severo.consumer;

import severo.molecule.Atom;
import severo.molecule.Molecule;
import storage.StorageItf;

public class ConsumerRacer{
	
	// MASTER MAIN
	public static void main(String[] args) throws Exception {
		
		StorageItf.init("/home/cdiaz/workspaceJ/severo/apps/COMPSs-SCO/DataClients/Consumer/cfgfiles/consumer.properties");		
		
		// Get number of molecules
		int n = Integer.parseInt(args[0]);
		System.out.println("[ConsumerRacer] Executing main loop with " + n + " molecules.");
		
		// Get number of steps
		int m = Integer.parseInt(args[1]);
		System.out.println("[ConsumerRacer] Executing main loop with " + m + " steps.");		
		
        for (int j = 0; j < m; j++) {					
			// Do molecules' offsets
			ConsumerRacer.doMoleculesOffsets(n, m);
			
        	// Compute center of mass
			ConsumerRacer.computeMoleculesCenterOfMass(n, m);
		}
		
		System.out.println("[ConsumerRacer] End of execution.");						
		
	}
	
	public static void doMoleculesOffsets(int n, int m) {
		for (int i = 1; i <= n; i++) {
			Molecule molecule = new Molecule("Molecule"+i);
			// A call to a COMPSs Task
			ConsumerRacer.doMoleculeOffsetX(molecule, m);
			// A call to a COMPSs Task
			ConsumerRacer.doMoleculeOffsetY(molecule, m);
			// A call to a COMPSs Task
			ConsumerRacer.doMoleculeOffsetZ(molecule, m);
		}
	}
	
	// This is a COMPSs Task
	public static void doMoleculeOffsetX(Molecule molecule, int m) {		
		Atom[] atoms = molecule.getAtoms();
		for (Atom a: atoms) {
			System.out.println("OLD: " + molecule.getName() + "." + a.getElementName() + ".X = " + a.getPoint().getX());
			if (a.getPoint().getX() < m) a.getPoint().setX(a.getPoint().getX() + 1);
			System.out.println("NEW: " + molecule.getName() + "." + a.getElementName() + ".X = " + a.getPoint().getX());
		}		
	}
	
	// This is a COMPSs Task
	public static void doMoleculeOffsetY(Molecule molecule, int m) {		
		Atom[] atoms = molecule.getAtoms();
		for (Atom a: atoms) {
			System.out.println("OLD: " + molecule.getName() + "." + a.getElementName() + ".Y = " + a.getPoint().getY());
			if (a.getPoint().getY() < m) a.getPoint().setY(a.getPoint().getY() + 1);
			System.out.println("NEW: " + molecule.getName() + "." + a.getElementName() + ".Y = " + a.getPoint().getY());
		}		
	}
	
	// This is a COMPSs Task
	public static void doMoleculeOffsetZ(Molecule molecule, int m) {		
		Atom[] atoms = molecule.getAtoms();
		for (Atom a: atoms) {
			System.out.println("OLD: " + molecule.getName() + "." + a.getElementName() + ".Z = " + a.getPoint().getZ());
			if (a.getPoint().getZ() < m) a.getPoint().setZ(a.getPoint().getZ() + 1);
			System.out.println("NEW: " + molecule.getName() + "." + a.getElementName() + ".Z = " + a.getPoint().getZ());
		}		
	}
		
	public static void computeMoleculesCenterOfMass(int n, int m) {
		for (int i = 1; i <= n; i++) {
			Molecule molecule = new Molecule("Molecule"+i);
			// A call to a COMPSs Task
			molecule.computeCenterOfMass();
		}		
	}

}