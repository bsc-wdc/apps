package multi;

import randomforest.RandomForest;


public class MultiRandomForest {

    private static final int DEFAULT_NUM_RF = 5;


    /**
     * Entry point.
     * 
     * @param args System arguments.
     */
    public static void main(String[] args) {
        int numSamples = (args.length >= 1) ? Integer.parseInt(args[0]) : RandomForest.DEFAULT_NUM_SAMPLES;
        int numFeatures = (args.length >= 2) ? Integer.parseInt(args[1]) : RandomForest.DEFAULT_NUM_FEATURES;
        int numClasses = (args.length >= 3) ? Integer.parseInt(args[2]) : RandomForest.DEFAULT_NUM_CLASSES;
        int numInformative = (args.length >= 4) ? Integer.parseInt(args[3]) : RandomForest.DEFAULT_NUM_INFORMATIVE;
        int numRedundant = (args.length >= 5) ? Integer.parseInt(args[4]) : RandomForest.DEFAULT_NUM_REDUNDANT;
        int numRepeated = (args.length >= 6) ? Integer.parseInt(args[5]) : RandomForest.DEFAULT_NUM_REPEATED;
        int numClustersPerClass =
            (args.length >= 7) ? Integer.parseInt(args[6]) : RandomForest.DEFAULT_NUM_CLUSTERS_PER_CLASS;
        boolean shuffle = (args.length >= 8) ? Boolean.parseBoolean(args[7]) : RandomForest.DEFAULT_SHUFFLE;
        long randomState = (args.length >= 9) ? Long.parseLong(args[8]) : RandomForest.DEFAULT_RANDOM_STATE;
        int numEstimators = (args.length >= 10) ? Integer.parseInt(args[9]) : RandomForest.DEFAULT_NUM_ESTIMATORS;
        int numModels = (args.length >= 11) ? Integer.parseInt(args[10]) : RandomForest.DEFAULT_NUM_MODELS;
        int numRF = (args.length >= 12) ? Integer.parseInt(args[11]) : DEFAULT_NUM_RF;

        System.out.println("Launching " + numRF + " RF algorithms");
        for (int i = 1; i <= numRF; ++i) {
            if (RandomForest.DEBUG) {
                System.out.println("- Launching " + i + "/" + numRF + "...");
            }
            RandomForest.generateRandomModelWithTest(numSamples, numFeatures, numClasses, numInformative, numRedundant,
                numRepeated, numClustersPerClass, shuffle, randomState, numEstimators, numModels);
        }
        System.out.println("DONE");
    }
}
