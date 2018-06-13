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

static Block *get_block(char *file, int M) {
	Block *result;
	FILE *fp;

	result = Block::init(M, 0.0);
	fp = fopen(file, "r");
	
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			fscanf(fp, "%lf ", &(result->data[i][j]));
		}
		fscanf(fp, " \n");
	}
	fclose(fp);
	
	return result;
}



static void write_block(Block *b, char *file, int M) {
	
	FILE *fp;

	fp = fopen(file , "w");
	
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			fprintf(fp, "%lf ", b->data[i][j]);
		}
		fprintf(fp, " \n");
	}
	fclose(fp);
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

#ifdef COMPSS_WORKER

void multiplyBlocks(char *f1, char *f2, char *f3, int M) {

        cout << "multiply GPU" << endl;
        Block *block1, *block2, *block3;

        block1 = get_block(f1, M);
        block2 = get_block(f2, M);
        block3 = get_block(f3, M);

        block1->multiplyGPU(*(block2), *(block3));

        write_block(block1, f1, M);

}


void multiplyBlocks_CPU(char *f1, char *f2, char *f3, int M) {

	cout << "multiply CPU" << endl;
	Block *block1, *block2, *block3;

	block1 = get_block(f1, M);
	block2 = get_block(f2, M);
	block3 = get_block(f3, M);

	block1->multiplyCPU(*(block2), *(block3));

	write_block(block1, f1, M);

}


void init_block(char *file, int bSize, double val) {
	cout << "init block " << file << endl;
        FILE *fp = fopen(file, "w");
        for (int i = 0; i < bSize; i++){
        	for (int j = 0; j < bSize; j++){
        		fprintf(fp, "%lf ", val);
        	}
        	fprintf(fp, " \n");
        }
	fclose(fp);
}

#endif

