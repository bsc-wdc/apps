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
package sparselu.arrays;


public class SparseLU {

    public static int N, M; // N: matrix size, M: block size

    private static double[][][] _A;


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
                            if (isNull(ii, jj)) {
                                    _A[ii][jj] = null;
                            } else {
                                    _A[ii][jj] = SparseLUImpl.initBlock(ii, jj, N, M);
                            }         
                    }
            }
    }

    /*private static double[] allocate_clean_block() {
            double[] b = new double[M*M];
            for (int i = 0; i < b.length; i++)
                    b[i] = 0.0;
            return b;
    }*/

	
	
    private static void printMatrix(double[][][] matrix, String name) {
        System.out.println("MATRIX " + name + ":");
        for (int i = 0; i < N; i++) {
                 for (int j = 0; j < N; j++) {
                          //blockToDisk(matrix[i][j], i, j, name);
                        SparseLUImpl.printBlock(matrix[i][j], M);                        
                 }
                 System.out.println("");
         }
    }

	/*private static void blockToDisk(double[] block, int i, int j, String name) {
        try {
                FileOutputStream fos = new FileOutputStream(name + "." + i + "." + j);

                if (block == null) fos.write("null".getBytes());
                else {
                        for (int k1 = 0; k1 < M; k1++) {
                                for (int k2 = 0; k2 < M; k2++) {
                                        String str = new Double(block[k1 * M + k2]).toString() + " ";
                                        fos.write(str.getBytes());
                                }
                                fos.write( "\n".getBytes() );
                        }
                }
                fos.close();
        }
        catch (Exception e) {
                e.printStackTrace();
        }
    }*/

	
    public static void main(String args[]) {
    	if ( args.length != 2 ) {
    		System.out.println("Usage: java SparseLU <matrix_dimension> <block_dimension\n");
    		return;
        }

        N = Integer.parseInt(args[0]);
        M = Integer.parseInt(args[1]);

        System.out.println("Running with the following parameters:");
        System.out.println("- N: " + N);
        System.out.println("- M: " + M);

        _A = new double[N][N][];

        genmat();

        printMatrix(_A, "Ain");

        for (int kk = 0; kk < N; kk++) {
        	SparseLUImpl.lu0(_A[kk][kk]);
        	for (int jj = kk+1; jj < N; jj++)
        		if (_A[kk][jj] != null) SparseLUImpl.fwd(_A[kk][kk], _A[kk][jj]);
            for (int ii = kk+1; ii < N; ii++) {
            	if (_A[ii][kk] != null) {
            		SparseLUImpl.bdiv(_A[kk][kk], _A[ii][kk]);
            		for (int jj = kk+1; jj < N; jj++) {
            			if (_A[kk][jj] != null) {
            				if (_A[ii][jj] == null)
            					//_A[ii][jj] = allocate_clean_block();
            					_A[ii][jj] = SparseLUImpl.bmodAlloc(_A[ii][kk], _A[kk][jj]);
            				else
            					SparseLUImpl.bmod(_A[ii][kk], _A[kk][jj], _A[ii][jj]);
            			}
            		}
            	}
            }
        }

        printMatrix(_A, "Aout");
    }

}
