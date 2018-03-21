/********************************************************************/
/* File:         Molecule.java										*/
/* Created:      21/11/2013										    */
/*                                                                  */
/* Author:       carlos.diaz@bsc.es                                 */
/*                                                                  */
/* Barcelona Supercomputing Center  								*/
/********************************************************************/
package severo.moleculeArray;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;

public class Molecule implements Serializable{

	private String name;
	private float[][] atoms;
	private float[] center;

	public Molecule() {

	}

	public Molecule(int i) {
		String name = "Molecule" + i;
		setName(name);
	}
	
	public void init(int n) {
        	this.center = getDummyCenter();
	        this.atoms = getDummyAtoms(n);

		/*
                try {
                        Thread.sleep(5000);
                } catch ( InterruptedException e ) {
                        System.out.println(e);
                }
		*/

	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		if (name == null) {
			throw new IllegalArgumentException("[Molecule]: ERROR, argument cannot be null");
		}
		this.name = name;
	}
				
	// Computes the center of mass of the molecule based on its atoms.
	public void computeCenterOfMass() {

		float sumX = 0;
		float sumY = 0;
		float sumZ = 0;
		float totalMass = 0;

		System.out.println("[Molecule]: Computing center of mass of molecule " + this.getName());		

		float[] atom;
		int n = this.atoms.length;
		System.out.println("[Molecule]: Number of atoms is " + n);
		for (int i = 0; i < n; ++i) {
			atom = this.atoms[i];
			sumX += atom[3] * atom[0];
			sumY += atom[3] * atom[1];
			sumZ += atom[3] * atom[2];
			totalMass += atom[3];
		}

		if (this.center == null) {
			System.err.println("[Molecule]: Center is null");
		} else {			
			this.center[0] = sumX / totalMass;
			this.center[1] = sumY / totalMass;
			this.center[2] = sumZ / totalMass;
			this.center[3] = totalMass;

			System.out.println("[Molecule]: X : " + this.center[0]);
			System.out.println("[Molecule]: Y : " + this.center[1]);
			System.out.println("[Molecule]: Z : " + this.center[2]);
			System.out.println("[Molecule]: Mass : " + this.center[3]);
		}
		
		/*
		try {
			Thread.sleep(5000);
		} catch ( InterruptedException e ) {
		  	System.out.println(e);
		}
		*/

		return;
	}	
	
	// Dummy method to generate atoms for a molecule	
	private float[][] getDummyAtoms(int nAtoms) {
		float[][] atoms = new float[nAtoms][];
		for (int i = 0; i < nAtoms; i++) {
			atoms[i] = new float[]{1, 1, 1, 100};
		}
		return atoms;
	}	
	
	private float[] getDummyCenter() {
		float[] center = new float[]{0, 0, 0, 0};
		return center;
	}

	private float[] getCenter() {
		return center;
	}
			
}
