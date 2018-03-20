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

package matmul.objects;


public class Matmul {
	
	private static int N;	// MSIZE
	private static int M;	// BSIZE

	private Block[][] A;
	private Block[][] B;
	private Block[][] C;

	
	public void run() {
		System.out.println("Running with the following parameters:");
		System.out.println("- N: " + N);
		System.out.println("- M: " + M);

		initMatrices();

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					C[i][j].multiplyAccumulative(A[i][k], B[k][j]);
				}
            }
		}
		
		/*printMatrix(A, "A");
		printMatrix(B, "B");
		printMatrix(C, "C");*/
	}

	private void initMatrices() {
		A = new Block[N][N];
		B = new Block[N][N];
		C = new Block[N][N];
		
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = Block.initBlock(M, 2.0);
				B[i][j] = Block.initBlock(M, 2.0);
				C[i][j] = Block.initBlock(M, 0.0);
			}
		}

		/*for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				A[i][j] = Block.initBlock(M, 2.0);

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				B[i][j] = Block.initBlock(M, 2.0);

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				C[i][j] = Block.initBlock(M, 0.0);*/
	}
	
	public void printMatrix(Block[][] matrix, String name) {
		System.out.println("MATRIX " + name);
		for (int i = 0; i < N; i++) {
			 for (int j = 0; j < N; j++) {
				 //matrix[i][j].blockToDisk(i, j);
				matrix[i][j].printBlock();
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

