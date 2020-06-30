package conway;

import conway.ConwayImpl;
import conway.State;

import es.bsc.compss.api.COMPSs;

public class Conway {
	public static int W;
	public static int L;
	
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
		int ITERATIONS = Integer.parseInt(args[2]);
		
		//Timming
		final long startTime = System.currentTimeMillis();

		//Initial values
		State initial_state = new State(W, L); //Random
		System.out.println("Initial matrix: ");
		initial_state.print();
		
		//Iterations
		State state_A = new State(initial_state);
		State state_B = new State(initial_state);
		
		//int val = ConwayImpl.update_cell(state_A, 1, 1);
		
		State aux;
		for (int t = 0; t < ITERATIONS; ++t) {
			aux = state_A;
			state_A = state_B;
			state_B = aux;
			
			for (int i = 0; i < W; ++i) {
				for (int j = 0; j < L; ++j) {
					int val = ConwayImpl.update_cell(state_A, i, j);
					state_B.set(i, j, val);
				}
				if(i%16 == 0)
					System.out.print("#");
			}
			System.out.println();
			COMPSs.barrier();
		}
		
		//Result
		System.out.println("Final matrix: ");
		state_B.print();
		
		//Timming
		final long endTime = System.currentTimeMillis();
		System.out.println("Total execution time: " + (endTime - startTime)/1000.0);
	}
} 