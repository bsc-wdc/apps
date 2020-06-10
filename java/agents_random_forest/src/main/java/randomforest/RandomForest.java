package randomforest;

import data.dataset.DoubleDataSet;
import data.tree.TreeFitConfig;
import data.dataset.IntegerDataSet;
import data.tree.Tree;

import es.bsc.compss.api.COMPSs;

import java.io.IOException;
import java.util.Random;

import randomforest.config.DataSetConfig;
import randomforest.config.FeaturesNumFilter;
import randomforest.config.FitConfig;


public class RandomForest {

    public static final boolean DEBUG = true;
    private static final boolean TIMERS = false;

    public static final int DEFAULT_NUM_SAMPLES = 30_000;
    public static final int DEFAULT_NUM_FEATURES = 40;
    public static final int DEFAULT_NUM_CLASSES = 200;
    public static final int DEFAULT_NUM_INFORMATIVE = 20;
    public static final int DEFAULT_NUM_REDUNDANT = 2;
    public static final int DEFAULT_NUM_REPEATED = 1;
    public static final int DEFAULT_NUM_CLUSTERS_PER_CLASS = 2;
    public static final boolean DEFAULT_SHUFFLE = true;
    public static final long DEFAULT_RANDOM_STATE = 0L;
    public static final int DEFAULT_NUM_ESTIMATORS = 1;
    public static final int DEFAULT_NUM_MODELS = 4;


    // Task
    public static IntegerDataSet randomSelection(int lowerBoundary, int upperBoundary, int numElements,
        long randomSeed) {

        IntegerDataSet ds = new IntegerDataSet(numElements, 1);
        ds.populateRandom(new Integer[][] { { lowerBoundary,
            upperBoundary } }, randomSeed);
        return ds;
    }

    private static RandomForestDataSet createDataSet(DataSetConfig datasetConfig) {
        int numSamples = datasetConfig.getNumSamples();
        int numFeatures = datasetConfig.getNumFeatures();
        int numClasses = datasetConfig.getNumClasses();

        if (DEBUG) {
            System.out.println("Creating dataset with " + numSamples + " cases with " + numFeatures + " features each");
        }
        RandomForestDataSet dataset = new RandomForestDataSet(numSamples, numFeatures);

        if (DEBUG) {
            System.out.println("Cases will be classified on " + numClasses + " classes");
        }
        try {
            dataset.populateRandom(datasetConfig);
        } catch (IOException e) {
            System.err.println("ERROR: Cannot populate dataset");
            e.printStackTrace();
            System.exit(1);
        } catch (InterruptedException e) {
            System.err.println("ERROR: Unexpected interrupt on dataset population");
            e.printStackTrace();
            System.exit(1);
        }

        return dataset;
    }

    /**
     * Trains the random forest classifier.
     *
     * @param samples Dataset with n samples of m features each
     * @param classification Dataset of n values of the corresponding class
     * @param config Configuration to fit the RandomForest model
     */
    private static Tree[] fit(DoubleDataSet samples, IntegerDataSet classification, FitConfig config) {
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

    private static void generateRandomModel(int numSamples, int numFeatures, int numClasses, int numInformative,
        int numRedundant, int numRepeated, int numClustersPerClass, boolean shuffle, long randomState,
        int numEstimators) {

        if (DEBUG) {
            System.out.println("Running model with arguments:");
            System.out.println("- NumSamples " + numSamples);
            System.out.println("- NumFeatures " + numFeatures);
            System.out.println("- NumClasses " + numClasses);
            System.out.println("- NumInformative " + numInformative);
            System.out.println("- NumRedundant " + numRedundant);
            System.out.println("- NumRepeated " + numRepeated);
            System.out.println("- NumClustersPerClass " + numClustersPerClass);
            System.out.println("- Shuffle? " + shuffle);
            System.out.println("- RandomState " + randomState);
            System.out.println("- NumEstimators " + numEstimators);
        }

        // Features filter
        if (DEBUG) {
            System.out.println("Setup features");
        }
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

        // Configurations
        if (DEBUG) {
            System.out.println("Instantiate DataSet and Fit configurations");
        }
        DataSetConfig datasetConfig = new DataSetConfig(numSamples, numFeatures, numClasses, numInformative,
            numRedundant, numClustersPerClass, numRepeated, shuffle, randomState);
        FitConfig rfConfig = new FitConfig(numEstimators, distrDepth, numCandidateFeat, maxDepth, randomState);
        rfConfig.print();

        // Training
        long timeStartDataset, timeEndDataset, timeStartTraining, timeEndTraining;

        if (DEBUG) {
            System.out.println("Creating training dataset...");
        }
        if (TIMERS) {
            timeStartDataset = System.currentTimeMillis();
        }
        RandomForestDataSet dataset = createDataSet(datasetConfig);
        if (TIMERS) {
            COMPSs.barrier();
            timeEndDataset = System.currentTimeMillis();
        }

        if (DEBUG) {
            System.out.println("Training...");
        }
        if (TIMERS) {
            timeStartTraining = System.currentTimeMillis();
        }
        Tree[] tree = fit(dataset.getSamples(), dataset.getClasses(), rfConfig);
        if (TIMERS) {
            COMPSs.barrier();
            timeEndTraining = System.currentTimeMillis();
        }

        // Print results
        System.out.println("Training tree at " + tree.hashCode());
        if (TIMERS) {
            System.out.println("TIME DataSet" + (timeEndDataset - timeStartDataset));
            System.out.println("TIME Training: " + (timeEndTraining - timeStartTraining));
            System.out.println("TIME Total: " + (timeEndTraining - timeStartDataset));

        }

        // De-register objects
        if (DEBUG) {
            System.out.println("Deregistering dataset from COMPSs...");
        }
        COMPSs.deregisterObject(dataset);
    }

    /**
     * Main method for testing and training.
     * 
     * @param numSamples Number of samples.
     * @param numFeatures Number of features.
     * @param numClasses Number of classes.
     * @param numInformative Number of informative.
     * @param numRedundant Number of redundancies.
     * @param numRepeated Number of repeated elements.
     * @param numClustersPerClass Number of clusters per class.
     * @param shuffle Whether to shuffle the dataset or not.
     * @param randomState Initial random seed.
     * @param numEstimators Number of estimators.
     * @param numModels Number of training models to run.
     */
    public static void generateRandomModelWithTest(int numSamples, int numFeatures, int numClasses, int numInformative,
        int numRedundant, int numRepeated, int numClustersPerClass, boolean shuffle, long randomState,
        int numEstimators, int numModels) {

        // Run test model
        if (DEBUG) {
            System.out.println("Run test model");
        }
        generateRandomModel(numSamples, numFeatures, numClasses, numInformative, numRedundant, numRepeated,
            numClustersPerClass, shuffle, (long) numEstimators, numEstimators);

        // Run models
        if (DEBUG) {
            System.out.println("Run models");
        }
        for (int i = 1; i <= numModels; ++i) {
            if (DEBUG) {
                System.out.println("Running model " + i + "/" + numModels);
            }
            generateRandomModel(numSamples, numFeatures, numClasses, numInformative, numRedundant, numRepeated,
                numClustersPerClass, shuffle, randomState, numEstimators);
        }
    }

    /**
     * Entry point.
     * 
     * @param args System arguments.
     */
    public static void main(String[] args) {
        int numSamples = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_NUM_SAMPLES;
        int numFeatures = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_NUM_FEATURES;
        int numClasses = (args.length >= 3) ? Integer.parseInt(args[2]) : DEFAULT_NUM_CLASSES;
        int numInformative = (args.length >= 4) ? Integer.parseInt(args[3]) : DEFAULT_NUM_INFORMATIVE;
        int numRedundant = (args.length >= 5) ? Integer.parseInt(args[4]) : DEFAULT_NUM_REDUNDANT;
        int numRepeated = (args.length >= 6) ? Integer.parseInt(args[5]) : DEFAULT_NUM_REPEATED;
        int numClustersPerClass = (args.length >= 7) ? Integer.parseInt(args[6]) : DEFAULT_NUM_CLUSTERS_PER_CLASS;
        boolean shuffle = (args.length >= 8) ? Boolean.parseBoolean(args[7]) : DEFAULT_SHUFFLE;
        long randomState = (args.length >= 9) ? Long.parseLong(args[8]) : DEFAULT_RANDOM_STATE;
        int numEstimators = (args.length >= 10) ? Integer.parseInt(args[9]) : DEFAULT_NUM_ESTIMATORS;
        int numModels = (args.length >= 11) ? Integer.parseInt(args[10]) : DEFAULT_NUM_MODELS;

        generateRandomModelWithTest(numSamples, numFeatures, numClasses, numInformative, numRedundant, numRepeated,
            numClustersPerClass, shuffle, randomState, numEstimators, numModels);
    }
}
