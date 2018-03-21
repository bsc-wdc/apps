/********************************************************************/
/* File:         Molecule.java										*/
/* Created:      21/11/2013										    */
/*                                                                  */
/* Author:       carlos.diaz@bsc.es                                 */
/*                                                                  */
/* Barcelona Supercomputing Center  								*/
/********************************************************************/
package severo.molecule;

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

	public String name;
	public Atom[] atoms;
	public Atom center;

	public Molecule() {

	}

	public Molecule(int i) {
		String name = "Molecule" + i;
		setName(name);

		//Atom center = getDummyCenter();
		//setCenter(center);

		//Atom[] atoms = getDummyAtoms();
		//setAtoms(atoms);
	}
	
    public void init(int n) {
        Atom center = getDummyCenter();
        setCenter(center);

        Atom[] atoms = getDummyAtoms(n);
        setAtoms(atoms);
    }

	public String getName() {
		return name;
	}

	public void setName(String name) {
		if (name == null) {
			throw new IllegalArgumentException(
					"[Molecule]: ERROR, argument cannot be null");
		}
		this.name = name;
	}

	public Atom[] getAtoms() {
		return atoms;
	}

	public void setAtoms(Atom[] atoms) {
		this.atoms = atoms;
	}

	public Atom getCenter() {
		return center;
	}

	public void setCenter(Atom center) {
		this.center = center;
	}
		
	/*
	public static ArrayList readFromFile(String db) {
		ArrayList<Molecule> molecules = new ArrayList<Molecule>();
		try {
			File file = new File(db);
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			String[] data;
			while ((line = br.readLine()) != null) {
				Molecule m = new Molecule(0);
				data = line.split(" ");
				int i = 0;
				m.setName(data[i]);
				for(int k = 0; k<3; k++) {
					m.getAtoms()[k].setElementName(data[++i]);
					m.getAtoms()[k].getPoint().setX(Float.parseFloat(data[++i]));
					m.getAtoms()[k].getPoint().setY(Float.parseFloat(data[++i]));
					m.getAtoms()[k].getPoint().setZ(Float.parseFloat(data[++i]));
					m.getAtoms()[k].setMass(Float.parseFloat(data[++i]));
				}
				m.getCenter().setElementName(data[++i]);
				m.getCenter().getPoint().setX(Float.parseFloat(data[++i]));
				m.getCenter().getPoint().setY(Float.parseFloat(data[++i]));
				m.getCenter().getPoint().setZ(Float.parseFloat(data[++i]));
				m.getCenter().setMass(Float.parseFloat(data[++i]));
				molecules.add(m);
			}
			br.close();
		} catch (IOException e) {
			System.out.println("[Molecule] " + e.getMessage());
		}
		return molecules;
	}

	public void writeToFile(String db) {
		try {
			File file = new File(db);
			FileOutputStream fop = new FileOutputStream(file, true);
			OutputStreamWriter osw = new OutputStreamWriter(fop);
			BufferedWriter bwriter = new BufferedWriter(osw);

			// Molecule (name)
			bwriter.write(this.getName());

			// Atoms (name and point coordinates)
			int n = this.getAtoms().length;
			for (int i = 0; i < n; i++) {
				bwriter.write(" ");
				bwriter.write(this.getAtoms()[i].getElementName());
				bwriter.write(" ");
				bwriter.write(String.valueOf(this.getAtoms()[i].getPoint()
						.getX()));
				bwriter.write(" ");
				bwriter.write(String.valueOf(this.getAtoms()[i].getPoint()
						.getY()));
				bwriter.write(" ");
				bwriter.write(String.valueOf(this.getAtoms()[i].getPoint()
						.getZ()));
				bwriter.write(" ");
				bwriter.write(String.valueOf(this.getAtoms()[i].getMass()));
			}

			// Center (name and point coordinates)
			bwriter.write(" ");
			bwriter.write(this.getCenter().getElementName());
			bwriter.write(" ");
			bwriter.write(String.valueOf(this.getCenter().getPoint().getX()));
			bwriter.write(" ");
			bwriter.write(String.valueOf(this.getCenter().getPoint().getY()));
			bwriter.write(" ");
			bwriter.write(String.valueOf(this.getCenter().getPoint().getZ()));
			bwriter.write(" ");
			bwriter.write(String.valueOf(this.getCenter().getMass()));

			bwriter.newLine();

			bwriter.close();
			osw.close();
			fop.close();
            System.out.println("[Molecule] " + name + " written to file");
		} catch (IOException e) {
			System.out.println("[Molecule] " + e.getMessage());
		}

	}
	*/
	
	// Returns a new molecule with the summation
	public Molecule add(Molecule m) {
		
		Molecule sum = new Molecule();
		
		int n1 = this.getAtoms().length;
		int n2 = m.getAtoms().length;
		
		Atom[] atoms = new Atom[n1+n2];

		for (int i=0; i<n1; i++)
			atoms[i] = this.getAtoms()[i];
		
		for (int i=n1; i<n1+n2; i++)
			atoms[i] = this.getAtoms()[i-n1];
				
		sum.setAtoms(atoms);
		
		sum.setName("Sum" + this.getName() + m.getName());
				
		return sum;
		
	}
			
	// Computes the center of mass of the molecule based on its atoms.
	public void computeCenterOfMass() {

		float sumX = 0;
		float sumY = 0;
		float sumZ = 0;
		float totalMass = 0;

		System.out.println("[Molecule]: Computing center of mass of molecule " + this.getName());		

		Atom atom;
		int n = this.getAtoms().length;
		System.out.println("[Molecule]: Number of atoms is " + n);
		for (int i = 0; i < n; ++i) {
			atom = this.getAtoms()[i];
			sumX += atom.getMass() * atom.getPoint().getX();
			sumY += atom.getMass() * atom.getPoint().getY();
			sumZ += atom.getMass() * atom.getPoint().getZ();
			totalMass += atom.getMass();
		}

		if (getCenter() == null) {
			System.err.println("[Molecule]: Center is null");
		} else {
			getCenter().setElementName("Center");
			getCenter().setMass(totalMass);
			getCenter().getPoint().setX(sumX / totalMass);
			getCenter().getPoint().setY(sumY / totalMass);
			getCenter().getPoint().setZ(sumZ / totalMass);

			System.out.println("[Molecule]: X : " + center.getPoint().getX());
			System.out.println("[Molecule]: Y : " + center.getPoint().getY());
			System.out.println("[Molecule]: Z : " + center.getPoint().getZ());
		}

		return;
	}
	

	// Prints molecule center of mass
	public void printCenterOfMass() {

		System.out.println("[Molecule]: Printing center of mass of molecule "
				+ this.getName());

		if (center == null) {
			System.err.println("[Molecule]: Center is null");
		} else {
			System.out.println("[Molecule]: X : " + center.getPoint().getX());
			System.out.println("[Molecule]: Y : " + center.getPoint().getY());
			System.out.println("[Molecule]: Z : " + center.getPoint().getZ());
		}

		return;
	}

	
	// Dummy method to generate atoms for a molecule	
	private Atom[] getDummyAtoms(int nAtoms) {
		Atom[] atoms = new Atom[nAtoms];
		for (int i = 0; i < nAtoms; i++) {
			int k = i + 1;
			atoms[i] = new Atom("atom" + k, 1, 1, 1, 100);
		}
		return atoms;
	}	
	
	/*
	private Atom[] getDummyAtoms() {
		Atom[] atoms = new Atom[3];
		for (int i = 0; i < 3; i++) {
			int k = i + 1;
			atoms[i] = new Atom("atom" + k, 1, 1, 1, 100);
		}
		return atoms;
	}
	*/

	private Atom getDummyCenter() {
		Atom center = new Atom("NULL", 0, 0, 0, 0);
		return center;
	}
			
}
