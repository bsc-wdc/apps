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

int main(int argc, char **argv)
{
	Matrix A;
	Matrix B;
	Matrix C;

	if (argc < 2) {
		//TODO: usage
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

		initMatrix(&B,N,M,val);
		initMatrix(&C,N,M,0.0);

		cout << "Waiting for initialization...\n";


		// compss_wait_on(A);
		compss_wait_on(B);
		compss_wait_on(C);


		cout << "Initialization ends...\n";

		C.multiply(A, B);

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
