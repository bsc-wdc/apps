package severo.consumer;

import severo.molecule.Molecule;
import storage.StorageItf;

public class Consumer {	

	// MASTER MAIN
	public static void main(String[] args) throws Exception {
		
		StorageItf.init("/home/cdiaz/workspaceJ/severo/apps/COMPSs-SCO/DataClients/Consumer/cfgfiles/consumer.properties");		
		
		// Computing Center of Mass
		int n = Integer.parseInt(args[0]);
		System.out.println("[Consumer] Executing main loop with " + n + " molecules.");

		System.out.println("[Consumer] Getting center ids...");	
		for (int i = 1; i <= n; i++) {
			Molecule molecule = new Molecule("Molecule" + i);
			if ( molecule.getCenter() != null )
				System.out.println("[Consumer] Center id " + molecule.getCenter().getID() + " for " + molecule.getName() + "." );
			else 
				System.out.println("[Consumer] Center is NULL for " + molecule.getName() + "." );
			molecule.printCenterOfMass();
		}	
		
		System.out.println("[Consumer] End of execution.");
		
	}
}
