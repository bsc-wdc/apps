package conway.blocks;

import conway.blocks.ConwayImpl;
import es.bsc.compss.api.COMPSs;


public class Conway {

    private static final boolean DEBUG = true;
    private static final double MS_TO_S = 1_000.0;


    private static void usage() {
        System.out.println("    Usage: simple <W, L, ITERATIONS, B_SIZE, A_FACTOR>");
    }

    private static Block[][] initMatrix(int widthNumBlocks, int lengthNumBlocks, int blockSize) {
        Block[][] res = new Block[widthNumBlocks][lengthNumBlocks];

        for (int i = 0; i < widthNumBlocks; ++i) {
            for (int j = 0; j < lengthNumBlocks; ++j) {
                res[i][j] = ConwayImpl.initBlock(blockSize);
            }
        }

        return res;
    }

    public static void main(String[] args) throws Exception {
        // Parse application arguments
        if (args.length != 5) {
            usage();
            throw new Exception("[ERROR] Incorrect number of parameters");
        }
        final int widthElements = Integer.parseInt(args[0]);
        final int lengthElements = Integer.parseInt(args[1]);
        final int numIterations = Integer.parseInt(args[2]);
        final int blockSize = Integer.parseInt(args[3]);
        final int aFactor = Integer.parseInt(args[4]);

        final int widthNumBlocks = widthElements / blockSize;
        final int lengthNumBlocks = lengthElements / blockSize;

        if (DEBUG) {
            System.out.println("Application parameters:");
            System.out.println("- Elements Width: " + widthElements);
            System.out.println("- Elements Length: " + lengthElements);
            System.out.println("- Num. Iterations: " + numIterations);
            System.out.println("- Block size: " + blockSize);
            System.out.println("- A factor: " + aFactor);
        }

        // Timing
        final long startTime = System.currentTimeMillis();

        // Initialize state
        Block[][] stateA = initMatrix(widthNumBlocks, lengthNumBlocks, blockSize);
        // Initialize swap state (only structure, blocks will be copied)
        Block[][] stateB = new Block[widthNumBlocks][lengthNumBlocks];

        // Iterations
        for (int iter = 0; iter < numIterations; ++iter) {
            if (DEBUG) {
                System.out.println("Running iteration " + iter);
            }

            // Swap states
            if (iter != 0) {
                if (DEBUG) {
                    System.out.println("- Swapping starting states...");
                }
                for (int i = 0; i < widthNumBlocks; ++i) {
                    for (int j = 0; j < lengthNumBlocks; ++j) {
                        stateA[i][j] = stateB[i][j];
                    }
                }
            }

            // Update blocks
            if (DEBUG) {
                System.out.println("- Updating block states...");
            }
            for (int i = 0; i < widthNumBlocks; ++i) {
                for (int j = 0; j < lengthNumBlocks; ++j) {
                    // Obtain input blocks
                    Block[][] supra = new Block[3][3];
                    for (int off_i = 0; off_i < 3; ++off_i) {
                        for (int off_j = 0; off_j < 3; ++off_j) {
                            int iState = (i + off_i - 1 + widthNumBlocks) % widthNumBlocks;
                            int jState = (j + off_j - 1 + lengthNumBlocks) % lengthNumBlocks;
                            supra[off_i][off_j] = stateA[iState][jState];
                        }
                    }

                    // Call Update
                    stateB[i][j] = ConwayImpl.updateBlock(supra[0][0], supra[0][1], supra[0][2], supra[1][0],
                        supra[1][1], supra[1][2], supra[2][0], supra[2][1], supra[2][2], aFactor, blockSize);
                }
            }
        }

        // Results
        if (DEBUG) {
            System.out.println("Results:");

            for (int i = 0; i < widthNumBlocks; ++i) {
                for (int j = 0; j < lengthNumBlocks; ++j) {
                    System.out.println("Block [" + i + "," + j + "] = " + stateB[i][j]);
                }
            }
        }

        // Timing
        COMPSs.barrier();
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / MS_TO_S + "s");
    }
}
