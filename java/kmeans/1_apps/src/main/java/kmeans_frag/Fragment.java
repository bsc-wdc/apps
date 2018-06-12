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
package kmeans_frag;

import java.io.Serializable;

/**
 * A class to encapsulate an input set of Points for use in the KMeans program.
 */
public class Fragment implements Serializable, Cloneable {
    private double[][] points;
   
    /* Default constructor */
    public Fragment() {
    }
    
    public Fragment(double[][] pts) {
        this.points = pts;
    }
    
    public Fragment(int dimX, int dimY) {
    	this.points = new double[dimX][];
    	for (int i = 0; i < dimX; ++i) {
            this.points[i] = new double[dimY];
    	}
    }
    
    public double[][] getPoints() {
    	return this.points;
    }
    
    public double getPoint (int v, int dim) {
    	return this.points[v][dim];
    }
    
    public double[] getVector(int v){
    	return this.points[v];
    }
    
    public void setPoint (double value, int vector, int dim) {
    	assert(vector >= 0 && vector < this.points.length && dim >= 0 && dim < this.points[0].length);
    	this.points[vector][dim] = value;
    }
    
    public void setPoints (double[][] points) {
    	this.points = points;
    }
    
    public int getVectors () {
    	return this.points.length;
    }
    
    public int getDimensions () {
    	return this.points[0].length;
    }
    
    public void print() {
    	for (int i = 0; i < points.length; ++i) {
            for (int j = 0 ; j < points[i].length; ++j) {
                System.out.print (points[i][j] + " ");
            }
            System.out.println("");
    	}
    }
    
    public Fragment clone () {
        Fragment f = new Fragment(this.points);    // Allocate  new frag
        return f;
    } 
}