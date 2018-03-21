package severo.producer;

import java.util.ArrayList;

import severo.molecule.Atom;
import severo.molecule.Molecule;
import storage.StorageItf;


public class ProducerRacer {
	
	public static boolean exit = false;
  
	// MASTER MAIN
	public static void main(String[] args) throws Exception {
		
		StorageItf.init("/home/cdiaz/workspaceJ/severo/apps/COMPSs-SCO/DataClients/Producer/cfgfiles/producer.properties");		
				
		// Delete previous database content
		System.out.println("[ProducerRacer] Deleting previous contents");
		//ProducerRacer.deletePreviousContents();

		// Generating initial molecules
		// Get number of molecules
		int n = Integer.parseInt(args[0]);
		System.out.println("[ProducerRacer] Executing main loop with " + n + " molecules.");
		
		// Get number of steps
		int m = Integer.parseInt(args[1]);
		System.out.println("[ProducerRacer] Executing main loop with " + m + " steps.");			
				
		ArrayList<Molecule> molecules = ProducerRacer.createInitialMolecules(n);
	
		// Saving molecules in the storage system
		ProducerRacer.saveMolecules(molecules);
		
		// Free local memory
		molecules.clear();
					
		while (!exit) {			
			// Checking exit conditions
			//exit = ProducerRacer.checkExitConditions(1, n, m);
			Thread.sleep(10000);
		}
				
		System.out.println("[ProducerRacer] End of execution.");
	}

	// Get the molecule from DB or create it if doesn't exists.
	/*
	public static void deletePreviousContents() {
		Molecule prototype = new Molecule();
		ArrayList result = prototype.queryByExample();

		if (result != null) {
			for (int i = 0; i < result.size(); i++) {
				Molecule molecule = (Molecule) result.get(i);
				molecule.deletePersistent(true);
			}			

		}
	}
	*/
	
	public static ArrayList<Molecule> createInitialMolecules(int n) {	
		System.out.println("[ProducerRacer] Running default constructor...");			
		ArrayList<Molecule> molecules = new ArrayList();
		
		for (int i = 1; i <= n; i++) {
			Molecule molecule = new Molecule(i);
			molecules.add(molecule);
			System.out.println("[ProducerRacer] " + molecule.getName() + " created.");			
		}
		
		return molecules;
	}
	
	public static void saveMolecules(ArrayList<Molecule> molecules) {
		System.out.println("[ProducerRacer] Running makePersistent method ...");		
		for (Molecule molecule: molecules) {			
			//molecule.makePersistent(molecule.getName());
			System.out.println("[ProducerRacer] " + molecule.getName() + " saved." );
		}
	}		
	
	/*
	public static boolean checkExitConditions(int start, int end, int m) {
		boolean exit = true;
		if (start != end) {			
			int start1 = start;
			int end1 = start + ((end - start) / 2);
			int start2 = end1 + 1;
			int end2 = end;			
			exit = ProducerRacer.checkExitConditions(start1, end1, m);
			if (exit == false) return exit;
			return ProducerRacer.checkExitConditions(start2, end2, m);		
		} else {
			// Call to a COMPSs Task
			Molecule molecule = new Molecule("Molecule"+start); 
			return ProducerRacer.checkExitCondition(molecule, m);
		}											
	}
	*/
	
	// This is a COMPSs Task
	public static boolean checkExitCondition(Molecule molecule, int m) {		
		Atom[] atoms = molecule.getAtoms();
		for (Atom a: atoms) {
			System.out.println(molecule.getName() + ".X = " +  a.getPoint().getX());
			if (a.getPoint().getX() < m) return false;
			System.out.println(molecule.getName() + ".Y = " +  a.getPoint().getY());
			if (a.getPoint().getY() < m) return false;
			System.out.println(molecule.getName() + ".Z = " +  a.getPoint().getZ());
			if (a.getPoint().getZ() < m) return false;
		}
		return true;
	}
	
}
