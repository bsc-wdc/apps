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
#include <sstream>
#include <unistd.h>


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
	Matrix mat;

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

		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){

				stringstream ss1, ss2, ss3;

        			ss1 << "A." << i << "." << j;
				ss2 << "B." << i << "." << j;
				ss3 << "C." << i << "." << j;

        			char * f1 = strdup(ss1.str().c_str());
				char * f2 = strdup(ss2.str().c_str());
				char * f3 = strdup(ss3.str().c_str());
				
				init_block(f1,M,val);
				init_block(f2,M,val);
				init_block(f3,M,0.0);		
			}
		}		


		cout << "Waiting for initialization...\n";

		cout << "Initialization ends...\n";

		for (int i=0; i<N; i++) {
        		for (int j=0; j<N; j++) {
            			for (int k=0; k<N; k++) {
					stringstream ss1, ss2, ss3;
					ss1 << "C." << i << "." << j;
					ss2 << "A." << i << "." << k;
					ss3 << "B." << k << "." << j;
					
					char * f1 = strdup(ss1.str().c_str());
                                	char * f2 = strdup(ss2.str().c_str());
                                	char * f3 = strdup(ss3.str().c_str());

					multiplyBlocks(f1, f2, f3, M);
            			}
            		}
        	}

// Uncomment this code chunk to check that every C file exists, check them in the files folder for results.
/*
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				stringstream ss;
				ss1 << "C." << i << "." << j;
				char * f = strdup(ss.str().c_str());
				FILE* file = compss_fopen(f,"r");
				fclose(file);
			}
		}
*/

		compss_off();

	}

	return 0;
}
