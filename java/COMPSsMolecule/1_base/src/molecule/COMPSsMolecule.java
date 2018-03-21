/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package molecule;

import java.io.Serializable;

import client.ClientDataServiceLib;

public class COMPSsMolecule implements Serializable {

	// Persistent Object
	private Molecule innerMolecule = null;

	public Molecule getInnerMolecule() {
		return innerMolecule;
	}

	public void setInnerMolecule(Molecule innerMolecule) {
		this.innerMolecule = innerMolecule;
	}

	// Default constructor needed for COMPSs
	public COMPSsMolecule() {
	}

	// COMPSsMolecule private constructor (use factory method instead)
	private COMPSsMolecule(Molecule innerMolecule) {
		this.innerMolecule = innerMolecule;
	}	
	
	
	// Factory method to create a new COMPSsMolecule
	public static COMPSsMolecule createCOMPSsMolecule(String name, Atom[] atoms) {		
		Molecule innerMolecule = new Molecule(name, atoms);		
		COMPSsMolecule compssMolecule = new COMPSsMolecule(innerMolecule);
		System.out.println("[COMPSsMolecule]: New molecule " + compssMolecule.innerMolecule.getName());
		compssMolecule.aggregateCenterOfMass();		
		return compssMolecule;
	}
	
	
	// Get the molecule from DB or create it if doesn't exists.
	public static COMPSsMolecule getMolecule(int index) {
		
		String name = "Molecule" + index;

		Molecule_Prototype prototype = new Molecule_Prototype(name, null, null, 0, 0, 0, 0);
		Molecule molecule = ClientDataServiceLib.queryByExampleFirst(prototype);
		
		COMPSsMolecule compssMolecule = null;
		if (molecule != null) { 
			compssMolecule = new COMPSsMolecule(molecule);
			System.out.println("[COMPSsMolecule]: Got molecule " + compssMolecule.innerMolecule.getName());
		} else {
			COMPSsMolecule.createCOMPSsMolecule(name, COMPSsMolecule.getDummyAtoms(index));
		}
		return compssMolecule;	
	}	

	// Dummy method to generate atoms for a molecule
	private static Atom[] getDummyAtoms(int nAtoms) {
		Atom[] atoms = new Atom[nAtoms];
		for (int i=0; i<nAtoms; i++) {
			int k = i+1;
			atoms[i] = new Atom("atom"+k,k,k,k,k*k);			
		}
		return atoms;
	}
	
	// Calculate the center of mass of the inner molecule.
	public Point getCenterOfMass() {

		float sumMX = 0;
		float sumMY = 0;
		float sumMZ = 0;
		float totalMass = 0;
		Atom atom;

		for (int i = 0; i < this.innerMolecule.getAtoms().length; ++i) {
			atom = this.innerMolecule.getAtoms()[i];
			sumMX += atom.getMass() * atom.getX();
			sumMY += atom.getMass() * atom.getY();
			sumMZ += atom.getMass() * atom.getZ();
			totalMass += atom.getMass();
		}

		return new Point(sumMX / totalMass, sumMY / totalMass, sumMZ
				/ totalMass, totalMass);

	}

	// Aggregate the center of mass of a molecule
	private void aggregateCenterOfMass() {

		// Variables used in calculation
		float centerX, centerY, centerZ;

		System.out.println("[COMPSsMolecule]: Calculating center of mass of molecule "
				+ this.innerMolecule.getName());

		// Calculate center of mass of molecule
		Point centerOfMass = this.innerMolecule.getCenterOfMass();

		// Aggregate result with variables
		this.innerMolecule.setSumX(centerOfMass.getX() * centerOfMass.getMass());
		this.innerMolecule.setSumY(centerOfMass.getY() * centerOfMass.getMass());
		this.innerMolecule.setSumZ(centerOfMass.getZ() * centerOfMass.getMass());
		this.innerMolecule.setSumMassOfMols(centerOfMass.getMass());

		System.out.println("[COMPSsMolecule]: Center of mass of molecule " + this.innerMolecule.getName());

		centerX = this.innerMolecule.getSumX() / this.innerMolecule.getSumMassOfMols();
		centerY = this.innerMolecule.getSumY() / this.innerMolecule.getSumMassOfMols();
		centerZ = this.innerMolecule.getSumZ() / this.innerMolecule.getSumMassOfMols();

		System.out.println("X = " + centerX);
		System.out.println("Y = " + centerY);
		System.out.println("Z = " + centerZ);

	}

	public void addMolecule(COMPSsMolecule compssMolecule) {
		// Variables used in calculation
		float centerX, centerY, centerZ;

		System.out.println("[COMPSsMolecule]: Adding " + compssMolecule.innerMolecule.getName() + " to " + this.innerMolecule.getName());

		// Aggregate result with variables
		this.innerMolecule.setSumX(this.innerMolecule.getSumX() + compssMolecule.innerMolecule.getSumX());
		this.innerMolecule.setSumY(this.innerMolecule.getSumY() + compssMolecule.innerMolecule.getSumY());
		this.innerMolecule.setSumZ(this.innerMolecule.getSumZ() + compssMolecule.innerMolecule.getSumZ());
		this.innerMolecule.setSumMassOfMols(this.innerMolecule.getSumMassOfMols() + compssMolecule.innerMolecule.getSumMassOfMols());

		System.out.println("[COMPSsMolecule]: Center of mass of molecule " + this.innerMolecule.getName());

		centerX = this.innerMolecule.getSumX() / this.innerMolecule.getSumMassOfMols();
		centerY = this.innerMolecule.getSumY() / this.innerMolecule.getSumMassOfMols();
		centerZ = this.innerMolecule.getSumZ() / this.innerMolecule.getSumMassOfMols();

		System.out.println("X = " + centerX);
		System.out.println("Y = " + centerY);
		System.out.println("Z = " + centerZ);
	}			
}
