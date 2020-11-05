package multi;

import kmeans.KMeans;


public class MultiKMeans {

    private static final int DEFAULT_NUM_KMEANS = 5;


    /**
     * Entry point.
     * 
     * @param args System arguments.
     */
    public static void main(String[] args) {
        final int numPoints = (args.length >= 1) ? Integer.valueOf(args[0]) : KMeans.DEFAULT_NUM_POINTS;
        final int numDims = (args.length >= 2) ? Integer.valueOf(args[1]) : KMeans.DEFAULT_NUM_DIMS;
        final int numFrags = (args.length >= 3) ? Integer.valueOf(args[2]) : KMeans.DEFAULT_NUM_FRAGS;
        final int numClusters = (args.length >= 4) ? Integer.valueOf(args[3]) : KMeans.DEFAULT_NUM_CLUSTERS;
        final int numIters = (args.length >= 5) ? Integer.valueOf(args[4]) : KMeans.DEFAULT_NUM_ITERS;
        final int numKmeans = (args.length >= 6) ? Integer.valueOf(args[5]) : DEFAULT_NUM_KMEANS;

        System.out.println("Launching " + numKmeans + " KMeans algorithms");
        for (int i = 1; i <= numKmeans; ++i) {
            if (KMeans.DEBUG) {
                System.out.println("- Launching " + i + "/" + numKmeans + "...");
            }
            KMeans.kmeans(numPoints, numDims, numFrags, numClusters, numIters);
        }
        System.out.println("DONE");

    }

}
