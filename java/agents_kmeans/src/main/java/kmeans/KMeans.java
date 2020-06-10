package kmeans;

import java.util.Random;
import java.lang.reflect.Array;


public class KMeans {

    public static final boolean DEBUG = true;

    public static final int DEFAULT_NUM_POINTS = 2_000;
    public static final int DEFAULT_NUM_DIMS = 2;
    public static final int DEFAULT_NUM_FRAGS = 2;
    public static final int DEFAULT_NUM_CLUSTERS = 4;
    public static final int DEFAULT_NUM_ITERS = 20;


    private static void initializePoints(KMeansDataSet data, int numFrags) {
        int pointsPerFragment = data.numPoints / numFrags;
        for (int i = 0; i < numFrags; i++) {
            int start = i * pointsPerFragment;
            int stop = Math.min(start + pointsPerFragment - 1, data.numPoints * data.numDimensions);
            int numPointsFrag = stop - start + 1;
            data.points[i] = initPointsFrag(numPointsFrag * data.numDimensions, i);
        }

        // Initialize cluster (copy first points)
        int nFrag = 0, startPos = 0;
        int toCopy = data.currentCluster.length;
        while (toCopy > 0) {
            int copied = copyToCluster(data.points[nFrag], data.currentCluster, toCopy, startPos);
            toCopy -= copied;
            startPos += copied;
            nFrag++;
        }
    }

    // Task
    public static float[] initPointsFrag(int size, int seed) {
        float[] points = new float[size];
        Random rnd = new Random(seed);
        for (int j = 0; j < size; j++) {
            points[j] = rnd.nextFloat();
        }

        return points;
    }

    private static int copyToCluster(float[] points, float[] cluster, int toCopy, int startPos) {
        int canCopy = Math.min(toCopy, Array.getLength(points));
        int j = 0;
        for (int i = startPos; i < startPos + canCopy; i++) {
            cluster[i] = points[j++];
        }
        return j;
    }

    // Task
    public static void computeNewLocalClusters(int myK, int numDimensions, float[] points, float[] clusterPoints,
        float[] newClusterPoints, int[] clusterCounts) {

        int numPoints = points.length / numDimensions;
        for (int pointNumber = 0; pointNumber < numPoints; pointNumber++) {
            int closest = -1;
            float closestDist = Float.MAX_VALUE;
            for (int k = 0; k < myK; k++) {
                float dist = 0;
                for (int dim = 0; dim < numDimensions; dim++) {
                    float tmp = points[pointNumber * numDimensions + dim] - clusterPoints[k * numDimensions + dim];
                    dist += tmp * tmp;
                }
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = k;
                }
            }

            for (int dim = 0; dim < numDimensions; dim++) {
                newClusterPoints[closest * numDimensions + dim] += points[pointNumber * numDimensions + dim];
            }
            clusterCounts[closest]++;
        }
    }

    // Task
    public static void accumulate(float[] onePoints, float[] otherPoints, int[] oneCounts, int[] otherCounts) {
        for (int i = 0; i < otherPoints.length; i++) {
            onePoints[i] += otherPoints[i];
        }
        for (int i = 0; i < otherCounts.length; i++) {
            oneCounts[i] += otherCounts[i];
        }
    }

    private static void localReduction(float[] points, int[] counts, int K, int numDimensions, float[] cluster) {
        for (int k = 0; k < K; k++) {
            float tmp = (float) counts[k];
            for (int dim = 0; dim < numDimensions; dim++) {
                points[k * numDimensions + dim] /= tmp;
            }
        }

        System.arraycopy(points, 0, cluster, 0, cluster.length);
    }

    /**
     * Main method for the KMeans algorithm.
     * 
     * @param numPoints Number of points.
     * @param numDims Number of dimensions per point.
     * @param numFrags Number of fragments.
     * @param numClusters Number of result clusters.
     * @param numIters Maximum number of iterations.
     */
    public static void kmeans(int numPoints, int numDims, int numFrags, int numClusters, int numIters) {
        if (DEBUG) {
            System.out.println("Running with the following parameters:");
            System.out.println("- Points: " + numPoints);
            System.out.println("- Dimensions: " + numDims);
            System.out.println("- Fragments: " + numFrags);
            System.out.println("- Clusters: " + numClusters);
            System.out.println("- Iterations: " + numIters);
        }

        // Generate dataset
        if (DEBUG) {
            System.out.println("Generating dataset");
        }
        KMeansDataSet data = KMeansDataSet.generateRandomPoints(numPoints, numDims, numFrags, numClusters);
        int[][] clusterCounts = new int[numFrags][numClusters];
        float[][] newClusters = new float[numFrags][numClusters * data.numDimensions];

        // Initialise points
        if (DEBUG) {
            System.out.println("Initialise points");
        }
        initializePoints(data, numFrags);

        // Do the requested number of iterations
        if (DEBUG) {
            System.out.println("Run iterations");
        }
        for (int iter = 1; iter <= numIters; iter++) {
            if (DEBUG) {
                System.out.println("- Iteration " + iter + "/" + numIters);
            }
            // Computation
            for (int i = 0; i < numFrags; i++) {
                float[] frag = data.points[i];
                computeNewLocalClusters(numClusters, numDims, frag, data.currentCluster, newClusters[i],
                    clusterCounts[i]);
            }

            // Reduction: points and counts
            // Stored in newClusters[0], clusterCounts[0]
            int size = newClusters.length;
            int i = 0, gap = 1;
            while (size > 1) {
                accumulate(newClusters[i], newClusters[i + gap], clusterCounts[i], clusterCounts[i + gap]);
                size--;
                i = i + 2 * gap;
                if (i == newClusters.length) {
                    gap *= 2;
                    i = 0;
                }
            }

            // Local reduction to get the new clusters
            // Adjust cluster coordinates by dividing each point value by the number of points in the cluster
            localReduction(newClusters[0], clusterCounts[0], numClusters, data.numDimensions, data.currentCluster);
        }

        // All done. Print the results
        System.out.println("KMeans DONE");
        if (DEBUG) {
            System.out.println("Result clusters: ");
            for (int i = 0; i < numClusters; i++) {
                for (int j = 0; j < data.numDimensions; j++) {
                    if (j > 0) {
                        System.out.print(" ");
                    }
                    System.out.print(data.currentCluster[i * data.numDimensions + j]);
                }
                System.out.println();
            }
            System.out.println();
        }
    }

    /**
     * Entry point.
     * 
     * @param args System arguments.
     */
    public static void main(String[] args) {
        final int numPoints = (args.length >= 1) ? Integer.valueOf(args[0]) : DEFAULT_NUM_POINTS;
        final int numDims = (args.length >= 2) ? Integer.valueOf(args[1]) : DEFAULT_NUM_DIMS;
        final int numFrags = (args.length >= 3) ? Integer.valueOf(args[2]) : DEFAULT_NUM_FRAGS;
        final int numClusters = (args.length >= 4) ? Integer.valueOf(args[3]) : DEFAULT_NUM_CLUSTERS;
        final int numIters = (args.length >= 5) ? Integer.valueOf(args[4]) : DEFAULT_NUM_ITERS;

        kmeans(numPoints, numDims, numFrags, numClusters, numIters);
    }

}
