package conway.blocks;

import conway.blocks.ConwayImpl;

import es.bsc.compss.api.COMPSs;


public class Conway {

    private static final double MS_TO_S = 1_000.0;

    protected static int WB;
    protected static int LB;
    protected static int B_SIZE;


    private static void usage() {
        System.out.println("    Usage: simple <W, L, ITERATIONS, B_SIZE>");
    }

    private static Block[][] initialiseBlock() {
        Block[][] res = new Block[WB][LB];

        for (int i = 0; i < WB; ++i) {
            for (int j = 0; j < LB; ++j) {
                res[i][j] = new Block();
            }
        }

        return res;
    }

    private static void swap(Block[][] stateA, Block[][] stateB) {
        Block c;
        for (int i = 0; i < WB; ++i) {
            for (int j = 0; j < LB; ++j) {
                c = stateA[i][j];
                stateA[i][j] = stateB[i][j];
                stateB[i][j] = c;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            usage();
            throw new Exception("[ERROR] Incorrect number of parameters");
        }

        int width = Integer.parseInt(args[0]);
        int length = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);
        B_SIZE = Integer.parseInt(args[3]);

        WB = width / Conway.B_SIZE;
        LB = length / Conway.B_SIZE;

        // Timming
        final long startTime = System.currentTimeMillis();

        // Initial values (Random)
        System.out.println("Initial matrix: ");
        // initial_state.print();

        // Iteration
        Block[][] stateA = initialiseBlock();
        Block[][] stateB = initialiseBlock();

        for (int t = 0; t < iterations; ++t) {
            swap(stateA, stateB);

            // Spawn tasks
            for (int i = 0; i < WB; ++i) {
                for (int j = 0; j < LB; ++j) {
                    Zone z = new Zone(stateA, i, j);
                    stateB[i][j] = ConwayImpl.updateBlock(z);
                }
            }

            COMPSs.barrier();
            System.out.println("#");
        }

        // Result
        System.out.println("Final matrix: ");
        // state_B.print();

        // Timming
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / MS_TO_S + "s");
    }

}
