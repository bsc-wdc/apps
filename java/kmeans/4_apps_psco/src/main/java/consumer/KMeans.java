package consumer;

import java.util.ArrayList;
import java.util.Arrays;

import model.Fragment;
import model.SumPoints;
import model.FragmentCollection;

import randomness.Randomness;

import storage.StorageItf;

// UNCOMMENT THIS IF YOU UNCOMMENT COMPSs.waitForAllTasks(); statements
import es.bsc.compss.api.COMPSs;


public class KMeans {

    private static int K = 4;                       // Number of clusters
    private static double epsilon = 1e-4;           // convergence criteria
    private static int iterations = 50;             // max iterations to converge
    private static String fragCollectionAlias = ""; // fragment collection
    private static boolean doDebug = false;

    private static String configPropertiesFile = null; // Only for DataClay
    

    public static void main(String args[]) throws Exception {
        checkArguments(args);

        if (configPropertiesFile != null) {
            StorageItf.init(configPropertiesFile);
        }

        FragmentCollection fragCollection = new FragmentCollection(0, fragCollectionAlias);
        System.out.println("[LOG] Obtained fragment collection: " + fragCollectionAlias + " ID " + fragCollection.getID() + " from "
                + fragCollection.getLocation());
        System.out.println("[LOG] Fragment collection has : " + fragCollection.getFragments().size() + " fragments. Total vectors "
                + fragCollection.getTotalVectors());

        // WARNING: Assuming all fragments have the same number of vectors and dimensions per vector
        int vectorsPerFragment = fragCollection.getVectorsPerFragment();
        int dimsPerVector = fragCollection.getNumDimensionsPerVector();
        System.out.println("[LOG] Running with the following parameters:");
        System.out.println("- Clusters      : " + K);
        System.out.println("- Iterations    : " + iterations);
        System.out.println("- VectorsPerFrag: " + vectorsPerFragment);
        System.out.println("- Dimensions    : " + dimsPerVector);

        // Mu generation
        // muResult = new Fragment(K, dimsPerVector);
        // muResult.fillPoints(Randomness.muSeed);
        // muResult.makePersistent(true, null);
        // System.out.println("[LOG] Random mu, 1st vector:" +
        // Arrays.toString(muResult.getVector(0)));

        // Run K-means
        System.out.println("[LOG] Computing result");
        long startTime = System.currentTimeMillis();
        computeKMeans(fragCollection);
        COMPSs.waitForAllTasks();
        long endTime = System.currentTimeMillis();
        System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");

        if (configPropertiesFile != null) {
            StorageItf.finish();
        }
    }

    private static void computeKMeans(FragmentCollection fc) {
        long init = 0, end;
        int dimensionsPerVector = fc.getNumDimensionsPerVector();
        int vectorsPerFragment = fc.getVectorsPerFragment();

        Fragment muResult = new Fragment(K, dimensionsPerVector);
        muResult.fillPoints(Randomness.muSeed);
        muResult.makePersistent(true, null);
        System.out.println("[LOG] Random mu");
        printKVectors(muResult);

        ArrayList<Fragment> fragments = fc.getFragments();
        int nFrags = fragments.size();

        Fragment oldmu = null;
        int currentIteration = 1;
        // Convergence condition
        while (currentIteration <= iterations && !Utils.has_converged(muResult, oldmu, epsilon)) {
            if (doDebug) {
                System.out.println("[LOG] Starting iteration " + currentIteration + " of " + iterations);
            }

            oldmu = muResult;
            // Clusters[] clusters = new Clusters[nFrags];
            SumPoints[] partialResult = new SumPoints[nFrags];
            int curFragment = 0;
            for (Fragment fragment : fragments) {
                // clusters[curFragment] = f.clusters_points_partial(mu, K,
                // vectorsPerFragment * curFragment);
                // partialResult[curFragment] =
                // f.partial_sum(clusters[curFragment], K, vectorsPerFragment *
                // curFragment);

                // MAP TASK
                if (doDebug) {
                    System.out.println("[LOG] mu id before map " + muResult.getID());
                    printKVectors(muResult);
                    System.out.println("[LOG] current fragment: " + fragment.getID());
                    printKVectors(fragment);
                    init = System.nanoTime();
                }
                partialResult[curFragment] = KMeansImpl.clusters_points_and_partial_sum(fragment, 
                                                                                        muResult,
                                                                                        K, 
                                                                                        vectorsPerFragment*curFragment
                                             );
                // partialResult[curFragment] = f.cluster_points_and_partial_sum(muResult, K, vectorsPerFragment * curFragment);
                if (doDebug) {
                    end = System.nanoTime();
                    if (partialResult[curFragment] == null) {
                        System.out.println("[PANIC] null partial result");
                        return;
                    }
                    System.out.println("[TIMER] MAP TASK : " + (end - init) / 1000 + " micros");
                    System.out.println("[LOG] mu id after map " + muResult.getID());
                }
                curFragment++;
            }

            COMPSs.waitForAllTasks();

            int neighbor = 1;
            while (neighbor < nFrags) {
                for (int i = 0; i < nFrags; i += 2 * neighbor) {
                    if (i + neighbor < nFrags) {
                        partialResult[i] = KMeansImpl.reduceCentersTask(partialResult[i], partialResult[i + neighbor]);
                    }
                }
                neighbor *= 2;
            }

            // NORMALIZE clusters
            if (doDebug) {
                System.out.println("[LOG] Normalize iteration " + currentIteration + " of " + iterations + ". mu " + muResult.getID());
            }
            muResult = new Fragment(partialResult[0].normalize());

            ++currentIteration;
        }

        System.out.println("[RESULT] Iterated " + (currentIteration - 1) + " times to get mu:");
        printKVectors(muResult);
    }

    private static void printKVectors(Fragment fragment) {
        System.out.println("[LOG] Printing " + K + " vectors of fragment with id " + fragment.getID());
        for (int i = 0; i < K; i++) {
            System.out.println("   - " + Arrays.toString(fragment.getVector(i)));
        }
    }

    private static void checkArguments(String[] args) {
        if (args.length < 1) {
            printUsage();
            System.exit(1);
        }
        if (args[0].equals("-h")) {
            printUsage();
            System.exit(0);
        } else {
            fragCollectionAlias = args[0];
        }

        for (int argIndex = 1; argIndex < args.length;) {
            String arg = args[argIndex++];
            if (arg.equals("-c")) {
                configPropertiesFile = args[argIndex++];
            } else if (arg.equals("-k")) {
                K = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-iterations")) {
                iterations = Integer.parseInt(args[argIndex++]);
            } else if (arg.equals("-debug")) {
                doDebug = true;
            } else if (arg.equals("-h")) {
                printUsage();
                System.exit(0);
            }
        }
    }

    private static void printUsage() {
        System.out.println("Usage \n\n" + KMeans.class.getName() + " <frag_col_alias> [-c <config_properties>] "
                + "[-k kvalue ] [-iterations iterations ] [-debug (debug info)] [-h (this help)] \n");
    }

}
