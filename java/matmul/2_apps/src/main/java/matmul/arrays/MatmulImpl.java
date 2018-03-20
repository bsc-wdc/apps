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

import java.io.FileOutputStream;


public class MatmulImpl {
	
	public static void multiplyAccumulative(double[] a, double[] b, double[] c) {
	
		int M = (int)Math.sqrt(a.length);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				for (int k = 0; k < M; k++) {
					c[i*M + j] += a[i*M + k] * b[k*M + j];
				}
			}
		}
	
	}
	
	public static double[] initBlock(int M, double initVal) {

		double[] block = new double[M*M];
		for (int k = 0; k < M*M; k++) {
			block[k] = initVal;
		}

		//try { Thread.sleep(200); } catch (Exception e) {}
	
		return block;
	}

	public static void printBlock(double[] block) {
		for (int k = 0; k < block.length; k++) {
			 System.out.print(block[k] + " ");
		 }
		System.out.println("");
	}
	
	public static void blockToDisk(double[] block, int i, int j, int M) {
		try {
			FileOutputStream fos = new FileOutputStream("C." + i + "." + j);
			
			for (int k1 = 0; k1 < M; k1++) {
				for (int k2 = 0; k2 < M; k2++) {
					String str = new Double(block[k1 * M + k2]).toString() + " ";
					fos.write(str.getBytes());
				}
				fos.write( "\n".getBytes() );
			}
			fos.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
