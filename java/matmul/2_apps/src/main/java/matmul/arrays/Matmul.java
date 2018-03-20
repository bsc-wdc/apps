/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package matmul.arrays;


public class Matmul {
	
	private static int N;	// MSIZE
	private static int M;	// BSIZE

	private double [][][]_A;
	private double [][][]_B;
	private double [][][]_C;

	
	public void run() {

		System.out.println("Running with the following parameters:");
		System.out.println("- N: " + N);
		System.out.println("- M: " + M);

		initMatrices();

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					MatmulImpl.multiplyAccumulative(_A[i][k], _B[k][j], _C[i][j]);
				}
            }
		}
		
		/*printMatrix(_A, "A");
		printMatrix(_B, "B");
		printMatrix(_C, "C");*/
	}

	private void initMatrices() {
		/* Hyper-matrices, where each block is represented by an array so that a whole block is allocated
		   contiguously in memory */
		_A = new double[N][N][];
		_B = new double[N][N][];
		_C = new double[N][N][];
		
		 for (int i = 0; i < N; i++) {
			 for (int j = 0; j < N; j++) {
				 _A[i][j] = MatmulImpl.initBlock(M, 2.0);
				 _B[i][j] = MatmulImpl.initBlock(M, 2.0);
				 _C[i][j] = MatmulImpl.initBlock(M, 0.0);
			 }
		 }

		/*for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) 
				_A[i][j] = MatmulImpl.initBlock(M, 2.0);

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				_B[i][j] = MatmulImpl.initBlock(M, 2.0);

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				_C[i][j] = MatmulImpl.initBlock(M, 0.0);*/
	}
	
	public void printMatrix(double[][][] matrix, String name) {
		System.out.println("MATRIX " + name);
		for (int i = 0; i < N; i++) {
			 for (int j = 0; j < N; j++) {
				 //MatmulImpl.blockToDisk(matrix[i][j], i, j, M);
				MatmulImpl.printBlock(matrix[i][j]);
			 }
			 System.out.println("");
		 }
	}
	
	public static void main(String args[]) {
		N = Integer.parseInt(args[0]);
		M = Integer.parseInt(args[1]);
		new Matmul().run();
	}
	
}

