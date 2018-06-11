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

import java.util.Random;

/**
 * KMeans fragments application.
 */
public class KMeans_frag {

    /**
     * Usage message.
     */
    private static void usage() {
        System.out.println("Usage: kmeans_frag.KMeans_frag");
        System.out.println("Available parameters:");
        System.out.println("\t -c <Clusters>      \t (Default: 4)");
        System.out.println("\t -i <Iterations>    \t (Default: 50)");
        System.out.println("\t -n <NumPoints>     \t (Default: 2000)");
        System.out.println("\t -d <Dimensions>    \t (Default: 2)");
        System.out.println("\t -f <NumFragments>  \t (Default: 2)");
        System.out.println("\t -s <Seed>          \t (Default: 5)");
        System.out.println("\t -r <ScaleFactor>   \t (Default: 1)");
        System.out.println("\t -ef <SameFragments>\t (Default: false)");
    }
    

    /**
     * Main function.
     * @param args List of arguments - see usage function.
     */
    public static void main(String[] args) {
    	// Default values
        int K = 4;			// k
        double epsilon = 1e-4;		// convergence criteria
        int iterations = 50;		// maxIterations
        int nPoints = 2000;		// numV
        int nDimensions = 2;            // Num dimesions
        int nFrags = 2;			// numFrag
        int seed = 5;                   // seed
        int scale = 1;                  // scale factor
        boolean sameFragments = false;  // generate all equal fragments
        
        // Get and parse arguments
        int argIndex = 0;
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
            } else if (arg.equals("-ef")) {
                sameFragments = Boolean.parseBoolean(args[argIndex++]);
            } else {
            	System.err.print("ERROR: Bad parameter: " + arg);
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
        System.out.println("- Same fragments: " + sameFragments);
        
        // KMeans execution
        long startTime = System.currentTimeMillis();
        computeKMeans(nPoints, K, epsilon, iterations, nFrags, nDimensions, seed, scale, sameFragments);
        long estimatedTime = System.currentTimeMillis() - startTime;
        // END
        System.out.println("-- END --");
        System.out.println("Elapsed time: " + estimatedTime);
    }
    
    
    /**
     * Compute KMeans function.
     * Performs the kmeans with the given parameters.
     * @param numV Number of vectors or points
     * @param k Number of clusters
     * @param epsilon Convergence value
     * @param maxIterations Max number of iterations
     * @param numFrag Amount of fragments
     * @param nDimensions Number of dimensions
     * @param seed Seed
     * @param scale Scale factor
     * @param sameFragments Equal or different fragments
     */
    private static void computeKMeans(int numV, int k, double epsilon, int maxIterations, int numFrag, int nDimensions, int seed, int scale, boolean sameFragments) {
    	System.out.println("--- Init computeKMeans ---");
    	Fragment[] dataSet = new Fragment[numFrag];
    	int sizeFrag = numV / numFrag;
        Random r = new Random(0);
        int rand = r.nextInt();
      
        // Random first centers
        Fragment mu = generateFragment(k, nDimensions, rand, scale);
    	System.out.println("--- Fragment MU created ---");
      
        // Generate fragments
        r = new Random(seed);
        if (sameFragments == true){
            // All fragments with the same random seed
            rand = r.nextInt();
            for(int i = 0; i < numFrag; i++) {
    		System.out.println("Fragment: " + i);
    		dataSet[i] = generateFragment(sizeFrag, nDimensions, rand, scale);   // @task
            }
        }else{
            // Each fragment with a different seed
            for(int i = 0; i < numFrag; i++) {
                System.out.println("Fragment: " + i);
                rand = r.nextInt();
                dataSet[i] = generateFragment(sizeFrag, nDimensions, rand, scale);   // @task
            }
        }
    	System.out.println("--- Fragments created ---");
    	
    	Fragment oldmu = null;
    	int n = 0;
    	System.out.println("--- Starting the iterations loop ---");
    	while ((n==0) || !Utils.has_converged(mu, oldmu, epsilon, n, maxIterations)) {
            System.out.println("--- Starting iteration " + n  + " ---");
            oldmu = mu;
            Clusters[] clusters = new Clusters[numFrag];       	 // key: cluster index - value: position list of the points 
            SumPoints[] partialResult = new SumPoints[numFrag];  // key: cluster index - value: tuple(numPointsCluster, sumOfAllClusterPoints)
            for (int f = 0; f < numFrag; f++){
                    clusters[f] = clusters_points_partial(dataSet[f], mu, k, sizeFrag*f);
                    partialResult[f] = partial_sum(dataSet[f], clusters[f], k, sizeFrag*f);
            }
            // MERGE-REDUCE                
            int neighbor=1;
            while (neighbor<numFrag){
                for (int i=0; i<numFrag; i+=2*neighbor){
                    if (i+neighbor < numFrag){
                        partialResult[i] = reduceCentersTask(partialResult[i], partialResult[i+neighbor]);
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
    
    
    /**
     * Function: generateFragment - @task.
     * Generates a fragment with the given parameters.
     * @param numV Number of vectors or points
     * @param nDimensions Amount of dimensions
     * @param seed Seed
     * @param scale Scale factor
     * @return Fragment considering the given parameters
     */
    public static Fragment generateFragment(int numV, int nDimensions, int seed, int scale) {
    	System.out.println("* Task Parameters:");
    	System.out.println("  - numV: " + numV);
    	System.out.println("  - nDimensions: " + nDimensions);
        System.out.println("  - seed: " + seed);
    	System.out.println("  - scale factor: " + scale);
    	double[][] points = new double[numV][nDimensions];
    	Random random = new Random(seed);
    	for(int i=0;i<numV;i++){
            for(int j=0;j<nDimensions;j++){
                // Random between [-scale, scale)
    		points[i][j] = (random.nextDouble() * (1 - (-1)) - 1) * scale;
            }
    	}
	return new Fragment(points);
    }
    

    /**
     * Function: clusters_points_partial - @task.
     * @param points Fragment of points
     * @param mu Fragment of centers
     * @param k Amount of clusters
     * @param ind Index
     * @return Clusters object.
     */
    public static Clusters clusters_points_partial(Fragment points, Fragment mu, int k, int ind) {
        Clusters clustersOfFrag = new Clusters(k);
        int numDimensions = points.getDimensions();
        for (int p = 0; p < points.getVectors(); p++){
            int closest = -1;
            double closestDist = Double.MAX_VALUE;
            for(int m = 0; m < mu.getVectors(); m++){
                double dist = 0;
                for (int dim = 0; dim < numDimensions; dim++) {
                    double tmp = points.getPoint(p, dim) - mu.getPoint(m, dim);
                    dist += tmp*tmp;
                }
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = m;          // Cluster to which belongs.
                }
            }
            int value = ind + p;
            clustersOfFrag.addIndex(closest, value);
        }
        return clustersOfFrag;
    }
    
    /**
     * Function: partial_sum - @task.
     * Calculates the partial sum for the given points agains the clusters
     * @param points Fragment of points or vectors
     * @param cluster Clusters
     * @param k Amount of clusters
     * @param ind Index
     * @return SumPoints object.
     */
    public static SumPoints partial_sum (Fragment points, Clusters cluster, int k, int ind) {
        SumPoints pSum = new SumPoints(k, points.getDimensions());
        for (int c = 0; c < cluster.getSize(); c++) {    // cluster.getSize == k
            int[] positionValues = cluster.getIndexes(c); 
            for (int i = 0; i < cluster.getIndexesSize(c); i++) {
                int value = positionValues[i];
                double[] v = points.getVector(value - ind);
                pSum.sumValue(c, v, 1);
            }
        }
        return pSum;
    }

    /**
     * Function: reduceCentersTask - @task
     * @param a First SumPoints object to reduce
     * @param b Second SumPoints object to reduce
     * @return A SumPoints object with the reduction of the two given.
     */
    public static SumPoints reduceCentersTask(SumPoints a, SumPoints b){
        for (int i = 0; i < b.getSize(); i++){
            a.sumValue(i, b.getValue(i), b.getNumPoints(i));
        }
        return a;
    }    
}

