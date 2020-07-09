package conway.accelerated;


import conway.accelerated.ConwayImpl;
//import es.bsc.compss.api.COMPSs;

public class Conway {

	private static final double MS_TO_S = 1_000.0;
	
	protected static int width;
	protected static int length;
	protected static int iterations;
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
		for (int i = 0; i < WB; ++i) {
			for (int j = 0; j < LB; ++j) {
				stateA[i][j] = stateB[i][j];
				stateB[i][j] = null;
			}
		}
	}
	
	private static void printState (Block[][] ref) {
		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < length; ++j) {
				System.out.print(ref[i/B_SIZE][j/B_SIZE].get(i%B_SIZE, j%B_SIZE));
			}
			System.out.println();
		}
		System.out.println();
	}
	
	private static int	sumState (Block[][] ref) {
		int sum = 0;
		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < length; ++j) {
				sum += ref[i/B_SIZE][j/B_SIZE].get(i%B_SIZE, j%B_SIZE);
			}
		}
		return sum;
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 5) {
			usage();
			throw new Exception("[ERROR] Incorrect number of parameters");
		}

		width = Integer.parseInt(args[0]);
		length = Integer.parseInt(args[1]);
		iterations = Integer.parseInt(args[2]);
		B_SIZE = Integer.parseInt(args[3]);
		A_FACTOR = Integer.parseInt(args[4]);

		WB = width / B_SIZE;
		LB = length / B_SIZE;
		
		// Timming
		final long startTime = System.currentTimeMillis();

		// Iteration
		Block[][] stateA = initialiseBlock();
		Block[][] stateB = initialiseBlock();
		
		//System.out.println("Initial Grid:");
		//printState(stateB);
		System.out.println("Initial sum: " + sumState(stateB));
		System.out.println("Iterating...");
		

		for (int t = 0; t < iterations / (A_FACTOR + 1); ++t) {
			swap(stateA, stateB);

			// Spawn tasks (for each block)
			for (int i = 0; i < WB; ++i) {
				for (int j = 0; j < LB; ++j) {
					
					//Obtain input blocks
					Block[][] supra =  new Block[3][3];
					
					for (int off_i = 0; off_i < 3; ++off_i) {
						for (int off_j = 0; off_j < 3; ++off_j) {
							int iState = (i + off_i - 1 + Conway.WB) % Conway.WB;
							int jState = (j + off_j - 1 + Conway.LB) % Conway.LB;
							supra[off_i][off_j] = stateA[iState][jState];
						}
					}
					
					//Call Update
					stateB[i][j] = ConwayImpl.updateBlock(supra[0][0], supra[0][1], supra[0][2],
												supra[1][0], supra[1][1], supra[1][2],
												supra[2][0], supra[2][1], supra[2][2],
												A_FACTOR, B_SIZE);
				}
			}
		}
		
		//Results
		//System.out.println("Final Grid:");
		
		for (int i = 0; i < WB; ++i) {
			for (int j = 0; j < LB; ++j) {
				System.out.print(stateB[i][j].get(0, 0));
			}
			System.out.println();
		}
		
		
		//printState(stateB);
		//System.out.println("Final sum: " + sumState(stateB));
		System.out.println("Iterating...");

		
		final long endTime = System.currentTimeMillis();
		System.out.println("Total execution time: " + (endTime - startTime) / MS_TO_S + "s");
	}

}
