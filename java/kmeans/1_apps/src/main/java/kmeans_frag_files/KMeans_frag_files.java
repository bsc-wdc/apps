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
package kmeans_frag_files;

import java.util.Random;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import kmeans_frag.Clusters;
import kmeans_frag.Fragment;
import kmeans_frag.SumPoints;
import kmeans_frag.Utils;

public class KMeans_frag_files {
	
    private static void usage() {
        System.out.println("Usage: kmeans_frag_files.KMeans_frag_files");
        System.out.println("Available parameters:");
        System.out.println("\t -c <Clusters>      \t (Default: 4)");
        System.out.println("\t -i <Iterations>    \t (Default: 50)");
        System.out.println("\t -n <NumPoints>     \t (Default: 2000)");
        System.out.println("\t -d <Dimensions>    \t (Default: 2)");
        System.out.println("\t -f <NumFragments>  \t (Default: 2)");
        System.out.println("\t -s <Seed>          \t (Default: 5)");
        System.out.println("\t -r <ScaleFactor>   \t (Default: 1)");
        System.out.println("\t -p <datasetPath>");
        System.out.println("\t -fc <firstCenters> \t (Default: 0 => random generated (Options: 1 => from random fragment, 2=> from random file))");
    }

    public static void main(String[] args) {
    	// Default values
        int K = 4;			// k
        double epsilon = 1e-4;		// convergence criteria
        int iterations = 50;		// maxIterations
        int nPoints = 2000;		// numV
        int nDimensions = 2;
        int nFrags = 2;			// numFrag
        int argIndex = 0;
        String datasetPath = "";	// dataset path
        int seed = 5;                   // Seed
        int scale = 1;                  // Scale factor
        int firstCenters = 0;           // 0 = random generated
                                        // 1 = from random fragment
                                        // 2 = from random file

        
	// Get and parse arguments
        while (argIndex < args.length) {
            String arg = args[argIndex++];
            if (arg.equals("-c")) {
                K = Integer.parseInt(args[argIndex++]);   
            } else if (arg.equals("-i")) {
                iterations = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-n")) {
            	nPoints = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-d")) {
            	nDimensions = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-f")) {
            	nFrags = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-s")) {
            	seed = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-r")) {
            	scale = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-p")) {
                datasetPath = args[argIndex++];
            } else if (arg.equals("-fc")) {
                firstCenters = Integer.parseInt(args[argIndex++]);
            } else {
            	// WARN: Disabled
            	System.err.print("ERROR: Bad parameter");
            	usage();
            	System.exit(1);
            }
        }

        System.out.println("-----------------------------------");
        System.out.println("KMeans with random generated points");
        System.out.println("-----------------------------------");
        System.out.println("Running with the following parameters:");
        System.out.println("- Clusters: " + K);
        System.out.println("- Iterations: " + iterations);
        System.out.println("- Points: " + nPoints);
        System.out.println("- Dimensions: " + nDimensions);
        System.out.println("- Fragments: " + nFrags);
        System.out.println("- Seed: " + seed);
        System.out.println("- Scale factor: " + scale);
        System.out.println("- Dataset path: " + datasetPath);
        System.out.println("- First centers: " + firstCenters);

        // Load data
        File path = new File(datasetPath);
        File [] files = path.listFiles();
        String[] fragments_files = new String[files.length];
        for (int i = 0; i < files.length; i++){
            if (files[i].isFile()){ //this line weeds out other directories/folders
            	fragments_files[i] = files[i].toString();
            }
        }
        
        // Check num files equal to given fragments
        assert(fragments_files.length == nFrags);
        
        // KMeans execution
        long startTime = System.currentTimeMillis();
        computeKMeans(fragments_files, nPoints, K, epsilon, iterations, nFrags, "random", nDimensions, seed, scale, firstCenters);
        long estimatedTime = System.currentTimeMillis() - startTime;
        
        // END
        System.out.println("-- END --");
        System.out.println("Elapsed time: " + estimatedTime);
    }
	
	
    private static void computeKMeans(String[] fragments_files, int numV, int k, double epsilon, int maxIterations, 
    	int numFrag, String initMode, int nDimensions, int seed, int scale, int firstCenters) {
    	
    	System.out.println("--- Init computeKMeans ---");
    	
    	// Read 1 file of data per task
    	Fragment[] dataSet = new Fragment[numFrag];
    	int sizeFrag = numV / numFrag;
    	for(int i = 0; i < numFrag; i++) {
    		System.out.println("Fragment: " + i);
    		dataSet[i] = readDatasetFromFile(fragments_files[i], sizeFrag, nDimensions);   // task
    	}
    	
    	System.out.println("--- Fragments Read ---");
    	
	// First centers
        Fragment mu = new Fragment();
        Random rand = new Random(seed);
        int ind;
        switch(firstCenters){
            case 0:
                // Random First Centers
    	        mu = generateFragment(k, nDimensions);
            case 1:
                // From Random Fragment
                ind = rand.nextInt(numFrag - 1);
                mu = init_random(dataSet[ind], k, nDimensions);
            case 2:
                // From Random File
                ind = rand.nextInt(numFrag - 1);
                mu = init_random(fragments_files[ind], k, nDimensions);
        }
        

    	System.out.println("--- Fragment MU created ---");
    	
    	Fragment oldmu = null;
    	int n = 0;
    	// Convergence condition
    	System.out.println("--- Starting the iterations loop ---");
    	while ((n==0) || !Utils.has_converged(mu, oldmu, epsilon, n, maxIterations)) {
            System.out.println("--- Starting iteration " + n  + " ---");
    		
            oldmu = mu;
            Clusters[] clusters = new Clusters[numFrag];        // key: cluster index - value: position list of the points 
            SumPoints[] partialResult = new SumPoints[numFrag]; // key: cluster index - value: tuple(numPointsCluster, sumOfAllClusterPoints)
            for (int f = 0; f < numFrag; f++){
    		clusters[f] = clusters_points_partial(dataSet[f], mu, k, sizeFrag*f);
    		partialResult[f] = partial_sum(dataSet[f], clusters[f], k, sizeFrag*f);
            }
            // MERGE-REDUCE                
            int neighbor=1;
            while (neighbor<numFrag){
                for (int i=0; i<numFrag; i+=2*neighbor){
                    if (i+neighbor < numFrag){
                        // with return
                        partialResult[i] = reduceCentersTask(partialResult[i], partialResult[i+neighbor]);
            		// with INOUT
            		//reduceCentersTask(partialResult[i], partialResult[i+neighbor]);
                    }
                }
            neighbor*=2;
            }
            // NORMALIZE clusters
            mu = partialResult[0].normalize(scale);
    	
            ++n;
    		
            System.out.println("--- finished iteration ---");
    	}
    }
    
    // @task
    public static Fragment readDatasetFromFile(String path, int numV, int nDimensions) {
    	System.out.println("* Task Parameters:");
    	System.out.println("  - PATH: " + path);
    	System.out.println("  - numV: " + numV);
    	System.out.println("  - nDimensions: " + nDimensions);
    	
    	double[][] points = new double[numV][nDimensions];
    	
    	FileReader fr = null;
    	BufferedReader br = null;
    	try {
    		fr = new FileReader(path);
    		br = new BufferedReader(fr);
    		
    	    int v = 0;
    	    while (v < numV) {
    	        String line = br.readLine();
    	        String[] values = line.split(" ");
    	        for (int i = 0; i < values.length; i++){
    	        	System.out.println("value of the point: " + values[i]);
    	        	points[v][i] = Double.valueOf(values[i]);
    	        }
    	        v = v + 1;
    	    }
    	} catch (IOException e){
    		e.printStackTrace();
    	} finally {
    		if (fr != null) {
    			try {
    				fr.close();
    			} catch (IOException e) {
    				e.printStackTrace();
    			}
    		}
    		if (br != null) {
    			try {
    				br.close();
    			} catch (IOException e) {
    				e.printStackTrace();
    			}
    		}
    	}
    	
    	return new Fragment(points);
    }
    
    // task
    public static Fragment generateFragment(int numV, int nDimensions) {
    	System.out.println("* Task Parameters:");
    	System.out.println("  - numV: " + numV);
    	System.out.println("  - nDimensions: " + nDimensions);
    	
    	double[][] points = new double[numV][nDimensions];
    	Random random = new Random();
    	
    	for(int i=0;i<numV;i++){
    		for(int j=0;j<nDimensions;j++){
    			// Random between [-1,1)
    			points[i][j] = random.nextDouble() * (1 - (-1)) - 1;
    		}
    	}
    	return new Fragment(points);
    }
    
    // task
    private static Fragment init_random(Fragment points, int k, int nDimensions){
	System.out.println("* Retrieving mu from fragment:");
    	System.out.println("  - k: " + k);
    	System.out.println("  - nDimensions: " + nDimensions);
        double[][] mu = new double[k][nDimensions];
    	int v = 0;
   	while (v < k) {
   	    for (int i = 0; i < nDimensions; i++){
   	     	mu[v][i] = points.getPoint(v, i);
   	    }
   	    v = v + 1;
   	}
    	return new Fragment(mu);
    }
    
    // task
    private static Fragment init_random(String file, int k, int nDimensions){
	System.out.println("* Retrieving mu from file:");
    	System.out.println("  - PATH: " + file);
    	System.out.println("  - k: " + k);
    	System.out.println("  - nDimensions: " + nDimensions);
        double[][] mu = new double[k][nDimensions];
    	
    	FileReader fr = null;
    	BufferedReader br = null;
    	try {
    		fr = new FileReader(file);
    		br = new BufferedReader(fr);
    		
    	    int v = 0;
    	    while (v < k) {
    	        String line = br.readLine();
    	        String[] values = line.split(" ");
    	        for (int i = 0; i < values.length; i++){
    	        	mu[v][i] = Float.valueOf(values[i]);
    	        }
    	        v = v + 1;
    	    }
    	} catch (IOException e){
    		e.printStackTrace();
    	} finally {
    		if (fr != null) {
    			try {
    				fr.close();
    			} catch (IOException e) {
    				e.printStackTrace();
    			}
    		}
    		if (br != null) {
    			try {
    				br.close();
    			} catch (IOException e) {
    				e.printStackTrace();
    			}
    		}
    	}
    	return new Fragment(mu);
    }

     
    // @task
    public static Clusters clusters_points_partial(Fragment points, Fragment mu, int k, int ind) {
        Clusters clustersOfFrag = new Clusters(k);
        int numDimensions = points.getDimensions();
        for (int p = 0; p < points.getVectors(); p++) {
            int closest = -1;
            double closestDist = Double.MAX_VALUE;
            for (int m = 0; m < mu.getVectors(); m++) {
                double dist = 0;
                for (int dim = 0; dim < numDimensions; dim++) {
                    double tmp = points.getPoint(p, dim) - mu.getPoint(m, dim);
                    dist += tmp * tmp;
                }
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = m; // belongs to this cluster
                }
            }
            int value = ind + p;
            clustersOfFrag.addIndex(closest, value);
        }
        return clustersOfFrag;
    }
	
    // @task
    public static SumPoints partial_sum(Fragment points, Clusters cluster, int k, int ind) {
        SumPoints pSum = new SumPoints(k, points.getDimensions());
        for (int c = 0; c < cluster.getSize(); c++) {  // en realidad cluster.getSize = k
            int[] positionValues = cluster.getIndexes(c);
            for (int i = 0; i < cluster.getIndexesSize(c); i++) {
                int value = positionValues[i];
                double[] v = points.getVector(value - ind);
                pSum.sumValue(c, v, 1);
            }
        }
        return pSum;
    }

    // with return
    // @task
    public static SumPoints reduceCentersTask(SumPoints a, SumPoints b) {
        for (int i = 0; i < b.getSize(); i++) {
            a.sumValue(i, b.getValue(i), b.getNumPoints(i));
        }

        return a;
    }

    /*
    // with INOUT
    // @task
    public static void reduceCentersTask(SumPoints a, SumPoints b){
            for (int i = 0; i < b.getSize(); i++){
                    a.sumValue(i, b.getValue(i), b.getNumPoints(i));
            }
    }
    */
	

    private static boolean has_converged (Fragment mu, Fragment oldmu, double epsilon, int n, int maxIterations){
        System.out.println("iter: " + n);
        System.out.println("maxIterations: " + maxIterations);

        if (oldmu == null) {
                return false;
        } else if (n >= maxIterations) {
                return true;
        } else {
            double aux = 0;
            for (int k = 0; k < mu.getVectors(); k++) {                     // loop over each center.
                    double dist = 0;
                    for (int dim = 0; dim < mu.getDimensions(); dim++) {    // loop over each center dimension.
                            double tmp = oldmu.getPoint(k, dim) - mu.getPoint(k, dim) ;
                            dist += tmp*tmp;
                    }
                    aux += dist;
            }
            if (aux < epsilon*epsilon) {
                    System.out.println("Distancia_T: " + aux);
                    return true;
            } else {
                    System.out.println("Distancia_F: " + aux);
                    return false;
            }
        }
    }   

    
    
}





