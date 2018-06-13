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
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <vector>

#define DEBUG_BINDING

#include "Matmul.h"
#include "Matrix.h"
#include "Block.h"

int N;  //MSIZE
int M;	//BSIZE
double val;

void usage() {
    cerr << "[ERROR] Bad number of parameters" << endl;
    cout << "    Usage: Matmul <N> <M> <val>" << endl;
}

int main(int argc, char **argv) {
	Matrix A;
	Matrix B;
	Matrix C;

	if (argc != 4) {
		usage();
		return -1;
	} else {
		N = atoi(argv[1]);
		M = atoi(argv[2]);
		val = atof(argv[3]);

		compss_on();

		cout << "Running with the following parameters:\n";
		cout << " - N: " << N << "\n";
		cout << " - M: " << M << "\n";
		cout << " - val: " << val << "\n";

		//initMatrix(&A,N,M,val);
		A = Matrix::init(N,M,val);
		B = Matrix::init(N,M,val);
		C = Matrix::init(N,M,0.0);
		//initMatrix(&B,N,M,val);
		//initMatrix(&C,N,M,0.0);

		cout << "Waiting for initialization...\n";


		//compss_wait_on(A);
		//compss_wait_on(B);
		//compss_wait_on(C);


		cout << "Initialization ends...\n";

		for (int i=0; i<N; i++) {
                	for (int j=0; j<N; j++) {

                        	for (int k=0; k<N; k++) {
					/*Block Aik = A.data[i][k];
					Block Bkj = B.data[k][j];
                                	C.data[i][j]->multiply(Aik, Bkj);*/
                                	multiplyBlocks(C.data[i][j], *A.data[i][k], *B.data[k][j]);
                        	}

                	}
        	}

        	for (int i=0; i<N; i++) {
                	for (int j=0; j<N; j++) {

                        	compss_wait_on(*C.data[i][j]);

                	}
        	}

		compss_off();

		if ((N <= 10) && (M <= 10)) {
			cout << "Matrix A\n";
			A.print();
			cout << "Matrix B\n";
			B.print();
			cout << "Matrix C\n";
			C.print();
		}

	}

	return 0;
}
