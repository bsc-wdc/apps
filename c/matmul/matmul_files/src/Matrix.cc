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

//#include <GS_templates.h>
#include <sstream>
#include <iostream>
#include "Matrix.h"

Matrix::Matrix(int mSize, int bSize, char mat_name) {
	N = mSize;
	M = bSize;
        matName = mat_name;	
	data.resize(N);
	for (int i=0; i<N; i++) {
		data[i].resize(N);
	}
}

void Matrix::init(int mSize, int bSize, double val, char mat_name) {
	Matrix matrix(mSize, bSize, mat_name);
	for (int i=0; i<mSize; i++) {
		for (int j=0; j<mSize; j++) {
			stringstream ss;
                        ss << mat_name << "." << i << "." << j;
                        std::string tmp = ss.str();
                
                        cout << "writting file " << tmp << endl; 
			FILE *fp = fopen(tmp.c_str(), "w");
			if (fp!=NULL){
				for (int ii = 0; ii < bSize; ii++){
					for (int jj = 0; jj < bSize; jj++){
						fprintf(fp, "%lf ", val);
					}
					fprintf(fp, " \n");
				}	
				fclose(fp);
			}else{
				cerr << " Error openning file " << tmp << " to write" <<endl;
				exit(1);
			}
		}
	}
}

void Matrix::print() {
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			data[i][j]->print();
			cout << "\r\n";
		}
		cout << "\r\n";
	}
}

