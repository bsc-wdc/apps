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

package sparselu.files;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;


public class SparseLU {

	public static int NB;
	
	private static boolean[][] matriu;


	private static void genmat() {
		boolean null_entry;

		for (int i = 0; i < NB; i++) 
			for (int j = 0; j < NB; j++) {
				matriu[i][j] = false;	
				null_entry = false;
				if ((i < j) && (i%3 != 0)) null_entry = true;
				if ((i > j) && (j%3 != 0)) null_entry = true;
				if (i%2 == 1) null_entry = true;
				if (j%2 == 1) null_entry = true;
				if (i == j) null_entry = false;
				if (i == j-1) null_entry = false;
				if (i-1 == j) null_entry = false;
				if (!null_entry)
					matriu[i][j] = true;
			}
	}

	public static void initialize(String filename, int ii, int jj) throws SparseLUAppException {
		FileOutputStream fos;
		try {
			fos = new FileOutputStream(filename);
		}
		catch (FileNotFoundException fnfe) {
			throw new SparseLUAppException(fnfe.getMessage());
		}
			
		try {
			double initVal = 1325;
			for ( int i = 0; i < Block.BLOCK_SIZE; i++ ) {	
				for(int j = 0; j < Block.BLOCK_SIZE; j++) {
					initVal = (3125 * initVal) % 65536;
					double cellValue = 0.0001;
					if (ii == jj) {
						if (i == j)
							cellValue = -20000;
						if ((i - 1 == j) || (i == j - 1))
							cellValue = 10000;
					}
					fos.write((cellValue + " ").getBytes());
				}
				fos.write("\n".getBytes());
			}
			fos.close();
		} catch (IOException ioe) {
			throw new SparseLUAppException(ioe.getMessage());
		}		
	}

	public static void main(String args[]) {
		String file, file1, file2, file3, file4;

		if ( args.length != 1 ) {
			System.out.println("Usage: java SparseLU <matrix_dimension>\n");
			return;
		}
		
		NB = Integer.parseInt(args[0]);
		
		matriu = new boolean[NB][NB];
		
		genmat();
		try {
			for (int i = 0; i < NB; i++)
				for (int j = 0; j < NB; j++)
					initialize("A." + i + "." + j, i, j);
			
			for (int k = 0; k < NB; k++) {
				file = "A." + k + "." + k;
				SparseLUImpl.lu0(file);
				for (int j = k+1; j < NB; j++) {
					if (matriu[k][j]) {
						file1 = "A." + k + "." + j;
						SparseLUImpl.fwd(file, file1); 
					}
				}
				for (int i = k+1; i < NB; i++) {
					if (matriu[i][k]) {
						file2 = "A." + i + "." + k;
						SparseLUImpl.bdiv(file, file2);
						for (int j = k+1; j < NB; j++) {
							if (matriu[k][j]) {
								file3 = "A." + k + "." + j;
								file4 = "A." + i + "." + j; 
								if (!matriu[i][j])
									matriu[i][j] = true;
								SparseLUImpl.bmod(file2, file3, file4);
							}
						}
					}
			    }
			}
		}
		catch (SparseLUAppException se) {
			se.printStackTrace();
			return;
		}
	}

}
