package severo.producer;

import java.util.ArrayList;

import severo.molecule.Molecule;
import storage.StorageItf;

public class Producer {
  
	// MASTER MAIN
	public static void main(String[] args) throws Exception {
		
		StorageItf.init("/home/cdiaz/workspaceS/severo1.3/apps/COMPSs-SCO/DataClients/Producer/cfgfiles/producer.properties");		
				
		// Delete previous database content
		//System.out.println("[Producer] Deleting previous contents");
		//Producer.deletePreviousContents();

		// Generating initial molecules
		int n = Integer.parseInt(args[0]);
		System.out.println("[Producer] Executing main loop with " + n + " molecules.");
		
		System.out.println("[Producer] Running default constructor...");			
		ArrayList<Molecule> molecules = new ArrayList();
		
		for (int i = 1; i <= n; i++) {
			Molecule molecule = new Molecule(i);
			molecules.add(molecule);
			System.out.println("[Producer] " + molecule.getName() + " created.");			
		}
				
		System.out.println("[Producer] Running makePersistent method ...");
		for (Molecule molecule: molecules) {			
			molecule.makePersistent(molecule.getName());
			System.out.println("[Producer] " + molecule.getName() + " saved." );
		}
		

		System.out.println("[Producer] Running computeCenterOfMass method...");	
		for (int i = 1; i <= n; i++) {
			Molecule molecule = molecules.get(i-1);
			
			molecule.computeCenterOfMass();
		}
		
		// Synchronization points!!!
		System.out.println("[Producer] Running printCenterOfMass method...");	
		for (int i = 1; i <= n; i++) {
			Molecule molecule = molecules.get(i-1);
			
			molecule.printCenterOfMass();
		}		
				
		molecules.clear();		
		System.out.println("[Producer] End of execution.");
	}

	/*
	// Get the molecule from DB or create it if doesn't exists.
	private static void deletePreviousContents() {
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
}
