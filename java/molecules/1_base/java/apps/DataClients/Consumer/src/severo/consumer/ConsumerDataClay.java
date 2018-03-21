package severo.consumer;

import severo.molecule.Molecule;
import storage.StorageItf;

public class ConsumerDataClay {	
	
	// MASTER MAIN
	public static void main(String[] args) throws Exception {
		
		StorageItf.init("/home/cdiaz/workspaceJ/severo/apps/COMPSs-SCO/DataClients/Consumer/cfgfiles/consumer.properties");		
		
		int n = Integer.parseInt(args[0]);
		System.out.println("[Consumer] Running with " + n + " molecules...");

		System.out.println("[Consumer] Ploting molecules...");
		for (int i = 1; i <= n; i++) {
			Molecule molecule = new Molecule("Molecule" + i);
			molecule.printCenterOfMass();
		}	
		
		System.out.println("[Consumer] End of execution.");
		
	}
}
