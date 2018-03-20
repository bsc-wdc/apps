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

package kmeans;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * A class to encapsulate an input set of 
 * Points for use in the KMeans program and the 
 * reading/writing of a set of points to data files.
 */
public class KMeansDataSet {
    private static final int cookie = 0x2badfdc0;
    private static final int version = 1;
    
    public final int numPoints;
    public final int numDimensions;
    public final float[][] points;
    public final float[] currentCluster;
    
    
    public KMeansDataSet(int np, int nd, float[][] pts, float[] cluster) {
        assert np * nd == pts.length;
        numPoints = np;
        numDimensions = nd;
        points = pts;
        currentCluster = cluster;
    }
    
    /*public final float getFloat(int point, int dim) {
        return points[point*numDimensions + dim];
    }*/
    
    public final int getPointOffset(int point) {
        return point*numDimensions;
    }
    
    
    /**
     * Create numPoints random points each of dimension numDimensions.
     */
    public static KMeansDataSet generateRandomPoints(int numPoints, int numDimensions, int numFrags, int K) {
        float[][] points = new float[numFrags][];
        float[] cluster = new float[K * numDimensions];
        
        return new KMeansDataSet(numPoints, numDimensions, points, cluster);
    }
    
    
    /**
     * Generate a set of random points and write them to a data file
     * @param fileName the name of the file to create
     * @param numPoints the number of points to write to the file
     * @param numDimensions the number of dimensions each point should have
     * @param seed a random number seed to generate the points.
     * @return <code>true</code> on success, <code>false</code> on failure
     */
    public static boolean generateRandomPointsToFile(String fileName, int numPoints, int numDimensions, int seed) {
    	DataOutputStream out = null;
        try {
            Random rand = new Random(seed);
            File outputFile = new File(fileName);
            out = new DataOutputStream(new FileOutputStream(outputFile));
            out.writeInt(cookie);
            out.writeInt(version);
            out.writeInt(numPoints);
            out.writeInt(numDimensions);
            int numFloats = numPoints * numDimensions;
            for (int i=0; i<numFloats; i++) {
                out.writeFloat(rand.nextFloat());
            }
        } catch (FileNotFoundException e) {
            System.out.println("Unable to open file for writing "+fileName);
            return false;
        } catch (IOException e) {
            System.out.println("Error writing data to "+fileName);
            e.printStackTrace();
            return false;
        } finally {
        	try {
				out.close();
			} catch (IOException e) {
				System.out.println("Error closing file " + fileName);
	            e.printStackTrace(); 
			}
        }
                
        return true;
    }
    
    
    /**
     * Write a set of points to a data file
     * @param fileName the name of the file to create
     * @param data the points to write
     * @return <code>true</code> on success, <code>false</code> on failure
     */
/*    public static boolean writePointsToFile(String fileName, KMeansDataSet data) {
        int numPoints = data.numPoints;
        if (numPoints == 0) return false;
        int numDimensions = data.numPoints;
        try {
            File outputFile = new File(fileName);
            DataOutputStream out = new DataOutputStream(new FileOutputStream(outputFile));
            out.writeInt(cookie);
            out.writeInt(version);
            out.writeInt(numPoints);
            out.writeInt(numDimensions);
            for (int i=0; i<numPoints*numDimensions; i++) {
                out.writeFloat(data.points[i]);
            }
        
        } catch (FileNotFoundException e) {
            System.out.println("Unable to open file for writing "+fileName);
            return false;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
                
        return true;
    }
*/    
    /**
     * Create numPoints random points each of dimension numDimensions.
     * @param fileName the name of the data file containing the points
     */
    /*public static KMeansDataSet readPointsFromFile(String fileName) {
        int i = 0;
        int j = 0;
        int numDimensions = 0;
        int numPoints = 0;
        float[] points = null;
        
        try {
            DataInputStream data = new DataInputStream(new FileInputStream(new File(fileName)));
            int fc = data.readInt();
            if (fc != cookie) {
                System.err.printf("Invalid cookie.  Found %d but expected %d\n", fc, cookie);
            }
            int fv = data.readInt();
            if (fv != version) {
                System.err.printf("Invalid version.  Found %d but expected %d\n", fc, cookie);
            }            
            numPoints = data.readInt();
            numDimensions = data.readInt();
            points = new float[numPoints*numDimensions];
            System.out.printf("Reading %d %d-dimensional points from %s\n", numPoints, numDimensions, fileName);
            for (i=0; i<numPoints; i++) {
                for (j=0; j<numDimensions; j++) {
                    points[i*numDimensions+ j] = data.readFloat();
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("Unable to open file "+fileName);
            System.exit(-1);
        } catch (IOException e) {
            System.err.printf("File did not contain enough data for %d %d-dimenstional points\n", numPoints, numDimensions);
            System.err.printf("Only found %d floats; expected to find %d\n", i*numDimensions+j, numPoints*numDimensions);
            e.printStackTrace();
            System.exit(-1);
        }
        
        return new KMeansDataSet(numPoints, numDimensions, points);
    }*/
	
}
