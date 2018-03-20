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

package sparselu.objects;


public class SparseLU {

	public static int N, M; // N: matrix size, M: block size

    private static Block[][] A;


    public static boolean isNull(int ii, int jj) {
            boolean nullEntry = false;
            if ((ii < jj) && (ii%3 != 0)) nullEntry = true;
            if ((ii > jj) && (jj%3 != 0)) nullEntry = true;
            if (ii%2 == 1)                nullEntry = true;
            if (jj%2 == 1)                nullEntry = true;
            if (ii == jj)                 nullEntry = false;
            if (ii == jj-1)               nullEntry = false;//
            if (ii-1 == jj)               nullEntry = false;//

            return nullEntry;
    }

    private static void genmat() {
        for (int ii = 0; ii < N; ii++) {
            for (int jj = 0; jj < N; jj++) {
                if (isNull(ii, jj))
                    A[ii][jj] = null;
                else
                    A[ii][jj] = Block.initBlock(ii, jj, N, M);
            }
        }
    }
	
	
    private static void printMatrix(Block[][] matrix, String name) {
        System.out.println("MATRIX " + name + ":");
        for (int i = 0; i < N; i++) {
        	for (int j = 0; j < N; j++) {
        		//matrix[i][j].blockToDisk(i, j, name);
        		if (matrix[i][j] == null)
        			System.out.println("null");
        		else
        			matrix[i][j].printBlock();
        	}
        }
	}

	
    public static void main(String args[]) {
    	if ( args.length != 2 ) {
    		System.out.println("Usage: java SparseLU <matrix_dimension> <block_dimension>\n");
    		return;
        }

        N = Integer.parseInt(args[0]);
        M = Integer.parseInt(args[1]);

        System.out.println("Running with the following parameters:");
        System.out.println("- N: " + N);
        System.out.println("- M: " + M);

        A = new Block[N][N];

        genmat();

        printMatrix(A, "Ain");

        for (int kk = 0; kk < N; kk++) {
        	A[kk][kk].lu0();
        	for (int jj = kk+1; jj < N; jj++)
        		if (A[kk][jj] != null)
        			A[kk][jj].fwd(A[kk][kk]);
            for (int ii = kk+1; ii < N; ii++) {
            	if (A[ii][kk] != null) {
            		A[ii][kk].bdiv(A[kk][kk]);
            		for (int jj = kk+1; jj < N; jj++) {
            			if (A[kk][jj] != null) {
            				if (A[ii][jj] == null)
            					A[ii][jj] = Block.bmodAlloc(A[ii][kk], A[kk][jj]);
            					//A[ii][jj] = new Block(M);
            				else
            				A[ii][jj].bmod(A[ii][kk], A[kk][jj]);
            			}
            		}
            	}
            }
        }

        printMatrix(A, "Aout");
    }
	
}
