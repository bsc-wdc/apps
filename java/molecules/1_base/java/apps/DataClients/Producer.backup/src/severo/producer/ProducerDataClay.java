package severo.producer;

import java.util.ArrayList;

import severo.molecule.Molecule;
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

            // Initialize
            System.out.println("[Producer] Initialize molecules (parallel)...");
            for (int i = 1; i <= nMols; i++) {
                Molecule molecule = molecules.get(i-1);
                molecule.init(nAtoms);
            }

            // Persist
            System.out.println("[Producer] Persist molecules (parallel)...");
            for (int i = 1; i <= nMols; i++) {
                Molecule molecule = molecules.get(i-1);
                molecule.makePersistent("Molecule" + i);
            }
            
            // Compute
            System.out.println("[Producer] Compute molecules (parallel)...");
            for (int i = 1; i <= nMols; i++) {
                Molecule molecule = molecules.get(i-1);
                molecule.computeCenterOfMass();
            }
            
            //Barrier
            System.out.println("[Producer] Barrier (sequential)...");
            for (int i = 1; i <= nMols; i++) {
                Molecule molecule = molecules.get(i-1);
                System.out.println("[Producer] " + molecule.getName() + " done!");
            }            

            molecules.clear();
            System.out.println("[Producer] End of execution.");
    }

}

