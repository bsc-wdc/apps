package conway.accelerated;

import conway.accelerated.Zone;
import conway.accelerated.ConwayImpl;
import es.bsc.compss.api.COMPSs;

public class Conway {

	private static final double MS_TO_S = 1_000.0;

	protected static int WB;
	protected static int LB;
	protected static int B_SIZE;
	protected static int A_FACTOR;

	private static void usage() {
		System.out.println("    Usage: simple <W, L, ITERATIONS, B_SIZE, A_FACTOR>");
	}

	private static Block[][] initialiseBlock() {
		Block[][] res = new Block[WB][LB];

		for (int i = 0; i < WB; ++i) {
			for (int j = 0; j < LB; ++j) {
				res[i][j] = new Block(B_SIZE);
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
		if (args.length != 5) {
			usage();
			throw new Exception("[ERROR] Incorrect number of parameters");
		}

		int width = Integer.parseInt(args[0]);
		int length = Integer.parseInt(args[1]);
		int iterations = Integer.parseInt(args[2]);
		B_SIZE = Integer.parseInt(args[3]);
		A_FACTOR = Integer.parseInt(args[4]);

		WB = width / Conway.B_SIZE;
		LB = length / Conway.B_SIZE;

		// Timming
		final long startTime = System.currentTimeMillis();

		// Iteration
		Block[][] stateA = initialiseBlock();
		Block[][] stateB = initialiseBlock();

		System.out.println("Iterating: ");

		for (int t = 0; t < iterations / (A_FACTOR + 1); ++t) {
			swap(stateA, stateB);

			// Spawn tasks
			for (int i = 0; i < WB; ++i) {
				for (int j = 0; j < LB; ++j) {
					Zone z = new Zone(stateA, i, j);
					Block res = new Block(B_SIZE);
					ConwayImpl.updateBlock(z, res, A_FACTOR);
					stateB[i][j] = res;
				}
			}
			

			COMPSs.barrier();
			System.out.print(".");
		}
		System.out.println();

		// Timming
		final long endTime = System.currentTimeMillis();
		System.out.println("Iterations ended");
		System.out.println("Total execution time: " + (endTime - startTime) / MS_TO_S + "s");
	}

}
