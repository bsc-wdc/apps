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

				cout << "creating iteration " << i << " " << j << endl;

				stringstream ss1, ss2, ss3;

        		ss1 << "files/A." << i << "." << j;
				ss2 << "files/B." << i << "." << j;
				ss3 << "files/C." << i << "." << j;

        		std::string tmp1 = ss1.str();
				std::string tmp2 = ss2.str();
				std::string tmp3 = ss3.str();

        		char * f1 = new char[tmp1.size() + 1];
				char * f2 = new char[tmp2.size() + 1];
				char * f3 = new char[tmp3.size() + 1];

        		std::copy(tmp1.begin(), tmp1.end(), f1);
				std::copy(tmp2.begin(), tmp2.end(), f2);
				std::copy(tmp3.begin(), tmp3.end(), f3);

        		f1[tmp1.size()] = '\0';
				f2[tmp2.size()] = '\0';
				f3[tmp3.size()] = '\0';

				cout << "cridant a init block " << f1 << endl;
				init_block(f1,M,val);
				cout << "cridant a init block " << f2 << endl;
				init_block(f2,M,val);
				cout << "cridant a init block " << f3 << endl;
				init_block(f3,M,0.0);		
			}
		}		


		cout << "Waiting for initialization...\n";

		cout << "Initialization ends...\n";

		for (int i=0; i<N; i++) {
        	for (int j=0; j<N; j++) {
            	for (int k=0; k<N; k++) {
					stringstream ss1, ss2, ss3;
					ss1 << "files/C." << i << "." << j;
					ss2 << "files/A." << i << "." << k;
					ss3 << "files/B." << k << "." << j;

					std::string tmp1 = ss1.str();
					std::string tmp2 = ss2.str();
					std::string tmp3 = ss3.str();

					char * f1 = new char[tmp1.size() + 1];
					std::copy(tmp1.begin(), tmp1.end(), f1);
					f1[tmp1.size()] = '\0';

                	char * f2 = new char[tmp2.size() + 1];
                	std::copy(tmp2.begin(), tmp2.end(), f2);
                	f2[tmp2.size()] = '\0';

                	char * f3 = new char[tmp3.size() + 1];
                	std::copy(tmp3.begin(), tmp3.end(), f3);
                	f3[tmp3.size()] = '\0';

					multiplyBlocks(f1, f2, f3, M);
            	}
            }
        }

// Uncomment this code chunk to check that every C file exists, check them in the files folder for results.
/*
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				stringstream ss;
				std::string tmp = ss.str();
                char * f = new char[tmp.size() + 1];
                std::copy(tmp.begin(), tmp.end(), f);
                f[tmp.size()] = '\0';
				FILE* file = compss_fopen(f,"r");
				fclose(file);
			}
		}
*/

		compss_off();

	}

	return 0;
}
