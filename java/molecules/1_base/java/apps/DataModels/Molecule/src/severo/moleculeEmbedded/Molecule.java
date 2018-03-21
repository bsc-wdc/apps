/********************************************************************/
/* File:         MoleculeEmbedded.java								*/
/* Created:      05/02/2015										    */
/*                                                                  */
/* Author:       anna.queralt@bsc.es                                */
/*                                                                  */
/* Barcelona Supercomputing Center  								*/
/********************************************************************/ 
package severo.moleculeEmbedded;

public class Molecule {
	
	public String name;
	//3 atoms, 1 point per atom
	public String atom1elementName;
	public float atom1mass;
	public float atom1PointX;
	public float atom1PointY;
	public float atom1PointZ;
	
	public String atom2elementName;
	public float atom2mass;
	public float atom2PointX;
	public float atom2PointY;
	public float atom2PointZ;

	public String atom3elementName;
	public float atom3mass;
	public float atom3PointX;
	public float atom3PointY;
	public float atom3PointZ;

	//Center of mass (atom)
	public String atomCelementName;
	public float atomCmass;
	public float atomCPointX;
	public float atomCPointY;
	public float atomCPointZ;

	
	public Molecule() {
		
	}
	
	public Molecule(int i) {
		String name = "Molecule" + i;
		setName(name);
		
		//Atom center = getDummyCenter();
		//setCenter(center);
		setDummyCenter();
		
		//Atom[] atoms = getDummyAtoms(nAtoms);
		//setAtoms(atoms);
		setDummyAtoms();
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
	
	public String getCenterName() {
		return this.atomCelementName;
	}
	
	//Center atom: setters and getters
	public void setCenterName(String name) {
		this.atomCelementName = name;
	}
	
	public void setCenterMass(float mass) {
		this.atomCmass = mass;
	}
	
	public float getCenterMass() {
		return this.atomCmass;
	}
	
	public void setCenterX(float x) {
		this.atomCPointX = x;
	}
	
	public void setCenterY(float y) {
		this.atomCPointY = y;
	}
	
	public void setCenterZ(float z) {
		this.atomCPointZ = z;
	}
	
	public float getCenterX() {
		return this.atomCPointX;
	}

	public float getCenterY() {
		return this.atomCPointY;
	}

	public float getCenterZ() {
		return this.atomCPointZ;
	}
	
	//3 atoms: setters and getters
	public void setAtom1Name(String name) {
		this.atom1elementName = name;
	}
	
	public String getAtom1Name() {
		return this.atom1elementName;
	}
	
	public void setAtom1Mass(float mass) {
		this.atom1mass = mass;
	}
	
	public float getAtom1Mass() {
		return this.atom1mass;
	}
	
	public void setAtom1X(float x) {
		this.atom1PointX = x;
	}
	
	public void setAtom1Y(float y) {
		this.atom1PointY = y;
	}
	
	public void setAtom1Z(float z) {
		this.atom1PointZ = z;
	}
	
	public float getAtom1X() {
		return this.atom1PointX;
	}
	
	public float getAtom1Y() {
		return this.atom1PointY;
	}

	public float getAtom1Z() {
		return this.atom1PointZ;
	}

	public void setAtom2Name(String name) {
		this.atom2elementName = name;
	}
	
	public String getAtom2Name() {
		return this.atom2elementName;
	}
	
	public void setAtom2Mass(float mass) {
		this.atom2mass = mass;
	}
	
	public float getAtom2Mass() {
		return this.atom2mass;
	}
	
	public void setAtom2X(float x) {
		this.atom2PointX = x;
	}
	
	public void setAtom2Y(float y) {
		this.atom2PointY = y;
	}
	
	public void setAtom2Z(float z) {
		this.atom2PointZ = z;
	}
	
	public float getAtom2X() {
		return this.atom2PointX;
	}
	
	public float getAtom2Y() {
		return this.atom2PointY;
	}

	public float getAtom2Z() {
		return this.atom2PointZ;
	}

	public void setAtom3Name(String name) {
		this.atom3elementName = name;
	}
	
	public String getAtom3Name() {
		return this.atom3elementName;
	}
	
	public void setAtom3Mass(float mass) {
		this.atom3mass = mass;
	}
	
	public float getAtom3Mass() {
		return this.atom3mass;
	}

	
	public void setAtom3X(float x) {
		this.atom3PointX = x;
	}
	
	public void setAtom3Y(float y) {
		this.atom3PointY = y;
	}
	
	public void setAtom3Z(float z) {
		this.atom3PointZ = z;
	}
	
	public float getAtom3X() {
		return this.atom3PointX;
	}
	
	public float getAtom3Y() {
		return this.atom3PointY;
	}

	public float getAtom3Z() {
		return this.atom3PointZ;
	}
	
	public void computeCenterOfMass() {

		float sumX = 0;
		float sumY = 0;
		float sumZ = 0;
		float totalMass = 0;

		System.out.println("[Molecule]: computing center of mass of molecule "
				+ this.getName());

		//Atom1
		sumX += this.getAtom1Mass() * this.getAtom1X();
		sumY += this.getAtom1Mass() * this.getAtom1Y();
		sumZ += this.getAtom1Mass() * this.getAtom1Z();
		totalMass += this.getAtom1Mass();
		//Atom2
		sumX += this.getAtom2Mass() * this.getAtom2X();
		sumY += this.getAtom2Mass() * this.getAtom2Y();
		sumZ += this.getAtom2Mass() * this.getAtom2Z();
		totalMass += this.getAtom2Mass();
		//Atom3
		sumX += this.getAtom3Mass() * this.getAtom3X();
		sumY += this.getAtom3Mass() * this.getAtom3Y();
		sumZ += this.getAtom3Mass() * this.getAtom3Z();
		totalMass += this.getAtom3Mass();

		if (this.getCenterName() == null) {
			System.err.println("[Molecule]: Center is null");
		} else {
			this.setCenterName("Center");
			this.setCenterMass(totalMass);
			this.setCenterX(sumX / totalMass);
			this.setCenterY(sumY / totalMass);
			this.setCenterZ(sumZ / totalMass);
			System.out.println("[Molecule]: X : " + this.getCenterX());
			System.out.println("[Molecule]: Y : " + this.getCenterY());
			System.out.println("[Molecule]: Z : " + this.getCenterZ());
		}
		
		return;
	}
	
	
	public void printCenterOfMass() {

		System.out.println("[Molecule]: Printing center of mass of molecule "
				+ this.getName());

		if (this.getCenterName() == null) {
			System.err.println("[Molecule]: Center is null");
		} else {
			System.out.println("[Molecule]: X : " + this.getCenterX());
			System.out.println("[Molecule]: Y : " + this.getCenterY());
			System.out.println("[Molecule]: Z : " + this.getCenterZ());
		}

		return;
	}

	
	private void setDummyAtoms() {
		setAtom1Name("atom1");
		setAtom1Mass(1*1);
		setAtom1X(1);
		setAtom1Y(1);
		setAtom1Z(1);
		
		setAtom1Name("atom2");
		setAtom1Mass(2*2);
		setAtom1X(2);
		setAtom1Y(2);
		setAtom1Z(2);

		setAtom1Name("atom3");
		setAtom1Mass(3*3);
		setAtom1X(3);
		setAtom1Y(3);
		setAtom1Z(3);
	}
	
	private void setDummyCenter() {
		setCenterName("NULL");
		setCenterMass(0);
		setCenterX(0);
		setCenterY(0);
		setCenterZ(0);
	}
	

}