/********************************************************************/
/* File:         Point.java											*/
/* Created:      21/11/2013										    */
/*                                                                  */
/* Author:       carlos.diaz@bsc.es                                 */
/*                                                                  */
/* Barcelona Supercomputing Center  								*/
/********************************************************************/ 
package severo.molecule;

import java.io.Serializable;

public class Point implements Serializable {

	public float x;
	public float y;
	public float z;

	public Point() {
	}
	
	public Point(float x, float y, float z) {
		this.x = x;
		this.y = y;
		this.z = z; 
	}

	public float getX() {
		return x;
	}

	public void setX(float x) {
		this.x = x;
	}
	
	public float getY() {
		return y;
	}

	public void setY(float y) {
		this.y = y;
	}

	public float getZ() {
		return z;
	}

	public void setZ(float z) {
		this.z = z;
	}	
}