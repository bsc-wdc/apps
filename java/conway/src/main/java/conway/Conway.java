package conway;

import conway.ConwayImpl;
import es.bsc.compss.api.COMPSs;

public class Conway {
	public static int WB;
	public static int LB;
	public static int B_SIZE;
	
	private static void usage() {
		System.out.println("    Usage: simple <W, L, ITERATIONS, B_SIZE>");
	}
	
	public static void main(String[] args) throws Exception {
		
		if (args.length != 4) {
			usage();
			throw new Exception("[ERROR] Incorrect number of parameters");
		}

		int W = Integer.parseInt(args[0]);
		int L = Integer.parseInt(args[1]);
		int ITERATIONS = Integer.parseInt(args[2]);
		B_SIZE = Integer.parseInt(args[3]);
		
		WB = W/Conway.B_SIZE;
		LB = L/Conway.B_SIZE;
		
	//Timming
		final long startTime = System.currentTimeMillis();

	//Initial values (Random)
		Base initial_state = new Base();
		System.out.println("Initial matrix: ");
		//initial_state.print();
		
		
	//Iterations
		Base state_A = new Base(initial_state);
		Base state_B = new Base(initial_state);

		for (int t = 0; t < ITERATIONS; ++t) {
			Base.swap(state_A, state_B);
			
			for (int I = 0; I < WB; ++I) {
				for (int J = 0; J < LB; ++J) {
					Zone Z = new Zone(state_A, I, J);
					Block T = new Block();
					ConwayImpl.update_block(Z, T);
					state_B.set(I, J, T);
				}
			}
			COMPSs.barrier();
			System.out.println("#");
		}
		
		//Result
		System.out.println("Final matrix: ");
		//state_B.print();
		
		//Timming
		final long endTime = System.currentTimeMillis();
		System.out.println("Total execution time: " + (endTime - startTime)/1000.0 + "sec");
		
	}
} 
