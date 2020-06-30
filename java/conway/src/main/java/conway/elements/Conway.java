package conway.elements;

import conway.elements.ConwayImpl;
import conway.elements.State;

import es.bsc.compss.api.COMPSs;


public class Conway {

    private static final double MS_TO_S = 1_000.0;

    protected static int W;
    protected static int L;


    private static void usage() {
        System.out.println("    Usage: simple <W, L, ITERATIONS>");
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            usage();
            throw new Exception("[ERROR] Incorrect number of parameters");
        }

        W = Integer.parseInt(args[0]);
        L = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);

        // Timming
        final long startTime = System.currentTimeMillis();

        // Initial values
        State initialState = new State(W, L); // Random
        System.out.println("Initial matrix: ");
        initialState.print();

        // Iterations
        State stateA = new State(initialState);
        State stateB = new State(initialState);

        // int val = ConwayImpl.update_cell(state_A, 1, 1);

        State aux;
        for (int t = 0; t < iterations; ++t) {
            aux = stateA;
            stateA = stateB;
            stateB = aux;

            for (int i = 0; i < W; ++i) {
                for (int j = 0; j < L; ++j) {
                    int val = ConwayImpl.updateCell(stateA, i, j);
                    stateB.set(i, j, val);
                }
                if (i % 16 == 0)
                    System.out.print("#");
            }
            System.out.println();
            COMPSs.barrier();
        }

        // Result
        System.out.println("Final matrix: ");
        stateB.print();

        // Timming
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / MS_TO_S + "s");
    }

}