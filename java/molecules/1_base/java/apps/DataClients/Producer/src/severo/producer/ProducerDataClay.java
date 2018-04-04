package severo.producer;

import java.util.ArrayList;

import severo.moleculeArray.Molecule;
import storage.StorageItf;

public class ProducerDataClay {

    // MASTER MAIN
    public static void main(String[] args) throws Exception {

            // Read arguments
            int nMols = Integer.parseInt(args[0]);
            int nAtoms = Integer.parseInt(args[1]);

            System.out.println("[Producer] Running with " + nMols + " molecules with " + nAtoms + " atoms...");

            // Create
            System.out.println("[Producer] Create molecules (sequential)...");
            ArrayList<Molecule> molecules = new ArrayList<Molecule>();
            for (int i = 1; i <= nMols; i++) {
                Molecule molecule = new Molecule(i);
                molecules.add(molecule);
            }
            
            // Persist
            System.out.println("[Producer] Persist molecules (parallel)...");
            int i = 1;
            for (Molecule molecule: molecules) {
		    molecule.makePersistent("Molecule" + i);
                    i = i++;
            }            

            // Initialize
            System.out.println("[Producer] Initialize molecules (parallel)...");
            for (Molecule molecule: molecules) {
                molecule.init(nAtoms);
            }
            
            // Compute
            System.out.println("[Producer] Compute molecules (parallel)...");
            for (Molecule molecule: molecules) {
                molecule.computeCenterOfMass();
            }
            
            //Barrier
            System.out.println("[Producer] Barrier (sequential)...");
            for (Molecule molecule: molecules) {              
		System.out.println("[Producer] " + molecule.getName() + " with center mass " + molecule.getCenter()[3]);
            }            

            molecules.clear();
            System.out.println("[Producer] End of execution.");
    }

}