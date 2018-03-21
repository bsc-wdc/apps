/********************************************************************/
/* File:         Atom.java											*/
/* Created:      21/11/2013										    */
/*                                                                  */
/* Author:       carlos.diaz@bsc.es                                 */
/*                                                                  */
/* Barcelona Supercomputing Center  								*/
/********************************************************************/ 
package severo.molecule;

import java.io.Serializable;


public class Atom implements Serializable{

	public Point point;
	public String elementName;
	public float mass;	

	public Atom(String elementName, float x, float y, float z, float mass) {
		this.point = new Point(x,y,z);
		this.setElementName(elementName);
		this.setMass(mass);
	}
	
	public Point getPoint() {
		return point;
	}

	public void setPoint(Point point) {
		this.point = point;
	}

	public String getElementName() {
		return elementName;
	}

	public void setElementName(String elementName) {
		if (elementName == null) { 
			throw new IllegalArgumentException("[Atom]: Argument cannot be null");
		}
		this.elementName = elementName;
	}

	public float getMass() {
		return mass;
	}

	public void setMass(float mass) {
		this.mass = mass;
	}
			
}
