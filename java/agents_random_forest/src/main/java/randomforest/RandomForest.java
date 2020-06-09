
package randomforest;

import data.dataset.DoubleDataSet;
import randomforest.config.DataSetConfig;
import randomforest.config.FitConfig;
import data.tree.TreeFitConfig;
import data.dataset.IntegerDataSet;
import data.tree.Tree;
import es.bsc.compss.api.COMPSs;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import randomforest.config.FeaturesNumFilter;


public class RandomForest {

    public static IntegerDataSet randomSelection(int lowerBoundary, int upperBoundary, int numElements,
        long randomSeed) {

        IntegerDataSet ds = new IntegerDataSet(numElements, 1);
        ds.populateRandom(new Integer[][] { { lowerBoundary,
            upperBoundary } }, randomSeed);
        return ds;
    }

    public static RandomForestDataSet createDataSet(int numSamples, int numFeatures, int numClasses, int numInformative,
        int numRedundant, int numClustersPerClass, int numRepeated, boolean shuffle, Long randomSeed)
        throws IOException, InterruptedException {

        DataSetConfig datasetConfig;
        datasetConfig = new DataSetConfig(numSamples, numFeatures, numClasses, numInformative, numRedundant,
            numClustersPerClass, numRepeated, shuffle, randomSeed);
        return createDataSet(datasetConfig);
    }

    public static RandomForestDataSet createDataSet(DataSetConfig datasetConfig)
        throws IOException, InterruptedException {

        int numSamples = datasetConfig.getNumSamples();
        int numFeatures = datasetConfig.getNumFeatures();
        int numClasses = datasetConfig.getNumClasses();
        System.out.println("Creating dataset with " + numSamples + " cases with " + numFeatures + " features each");
        RandomForestDataSet dataset = new RandomForestDataSet(numSamples, numFeatures);
        System.out.println("Cases will be classified on " + numClasses + " classes");
        dataset.populateRandom(datasetConfig);
        return dataset;
    }

    /**
     * Trains the random forest classifier.
     *
     * @param samples Dataset with n samples of m features each
     * @param classification Dataset of n values of the corresponding class
     * @param config Configuration to fit the RandomForest model
     */
    public static Tree[] fit(DoubleDataSet samples, IntegerDataSet classification, FitConfig config) {
        Tree[] trees = new Tree[config.getNumEstimators()];
        TreeFitConfig treeFitConfig = config.getTreeFitConfig();
        Random rd;
        rd = new Random();
        if (config.getRandomSeed() != null) {
            rd.setSeed(config.getRandomSeed());
        }

        int numSamples = samples.getNumSamples();

        for (int estimatorId = 0; estimatorId < config.getNumEstimators(); estimatorId++) {
            long randomSeed = rd.nextLong();
            IntegerDataSet sampleSelection = randomSelection(0, numSamples, numSamples, randomSeed);
            trees[estimatorId] =
                Tree.trainTreeWithDataset(samples, classification, sampleSelection, treeFitConfig, randomSeed);
            COMPSs.deregisterObject(sampleSelection);
        }
        for (int estimatorId = 0; estimatorId < config.getNumEstimators(); estimatorId++) {
            COMPSs.deregisterObject(trees[estimatorId]);
        }
        return trees;
    }

    /**
     * Returns the mean accuracy on the given test data.
     *
     * @param features array of n samples with m features each
     * @param classification array of n values with the corresponding class
     */
    public void score(Object[] features, Object[] classification) {
    }

    public static void generateRandomModelWithTest(String[] args) throws Exception {
        String[] argsTest = new String[args.length];
        System.arraycopy(args, 0, argsTest, 0, args.length);
        argsTest[args.length - 2] = argsTest[args.length - 1];

        generateRandomModel(argsTest);
        generateRandomModel(args);
        generateRandomModel(args);
        generateRandomModel(args);
        generateRandomModel(args);
    }

    public static void generateRandomModel(String[] args) throws Exception {
        System.out.println(Arrays.toString(args));
        // Data set generation parameters
        /*
         * int numSamples = 30_000; int numFeatures = 40; int numClasses = 200; int numInformative = 20; int
         * numRedundant = 2; int numRepeated = 1; int numClustersPerClass = 2; boolean shuffle = true;
         */
        int numSamples = Integer.parseInt(args[0]);
        int numFeatures = Integer.parseInt(args[1]);
        int numClasses = Integer.parseInt(args[2]);
        int numInformative = Integer.parseInt(args[3]);
        int numRedundant = Integer.parseInt(args[4]);
        int numRepeated = Integer.parseInt(args[5]);
        int numClustersPerClass = Integer.parseInt(args[6]);
        boolean shuffle = Boolean.parseBoolean(args[7]);

        Long randomState = Long.parseLong(args[8]);
        // Parse options
        int numEstimators = Integer.parseInt(args[9]);

        String tryFeatures = "sqrt";
        FeaturesNumFilter.Config featuresFilter = FeaturesNumFilter.Config.valueOf(tryFeatures.toUpperCase());
        int numCandidateFeat = FeaturesNumFilter.resolveNumCandidateFeatures(numFeatures, featuresFilter);

        int maxDepth = Integer.MAX_VALUE;
        String distrDepthConf = "auto";
        int distrDepth = 0;
        if (distrDepthConf.compareTo("auto") == 0) {
            distrDepth = Math.max(0, (int) (Math.log10(numSamples)) - 4);
            distrDepth = Math.min(distrDepth, maxDepth);
        } else {
            distrDepth = Integer.parseInt(distrDepthConf);
        }

        DataSetConfig datasetConfig;

        datasetConfig = new DataSetConfig(numSamples, numFeatures, numClasses, numInformative, numRedundant,
            numClustersPerClass, numRepeated, shuffle, randomState);

        FitConfig rfConfig;
        rfConfig = new FitConfig(numEstimators, distrDepth, numCandidateFeat, maxDepth, randomState);

        rfConfig.print();
        System.out.println("Training Classifier...");
        COMPSs.barrier();
        RandomForestDataSet dataset = createDataSet(datasetConfig);
        COMPSs.barrier();
        long startTime = System.currentTimeMillis();
        System.out.println("\tTraining starts at " + startTime);
        Tree[] tree = fit(dataset.getSamples(), dataset.getClasses(), rfConfig);
        COMPSs.barrier();
        long endTime = System.currentTimeMillis();
        System.out.println("Training tree at " + tree.hashCode());
        System.out.println("Training completed at " + endTime);
        System.out.println("Training length: " + (endTime - startTime));
        COMPSs.deregisterObject(dataset);

    }

    public static void main(String[] args) throws Exception {
        // String numSamples = 30_000 + "";
        // String numFeatures = 40 + "";
        // String numClasses = 200 + "";
        // String numInformative = 20 + "";
        // String numRedundant = 2 + "";
        // String numRepeated = 1 + "";
        // String numClustersPerClass = 2 + "";
        // String shuffle = true + "";
        // String randomSeed = "0";
        // String numEstimators = "1";
        // generateRandomModel(new String[]{numSamples, numFeatures, numClasses, numInformative, numRedundant,
        // numRepeated, numClustersPerClass, shuffle, randomSeed, numEstimators});

        generateRandomModelWithTest(args);
    }
}
