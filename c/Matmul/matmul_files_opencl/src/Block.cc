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

#include <iostream>
#include "Block.h"

Block::Block(int bSize) {
	M = bSize;
	data.resize(M);
	for (int i=0; i<M; i++) {
		data[i].resize(M);
	}
}

Block *Block::init(int bSize, double initVal) {
	Block *block = new Block(bSize);
	for (int i=0; i<bSize; i++) {
		for (int j=0; j<bSize; j++) {
			block->data[i][j] = initVal;
		}
	}
	return block;
}

#ifdef COMPSS_WORKER

void Block::multiplyCPU(Block block1, Block block2) {
	for (int i=0; i<M; i++) {
		for (int j=0; j<M; j++) {
			for (int k=0; k<M; k++) {
				data[i][j] += block1.data[i][k] * block2.data[k][j];
			}
		}
	}
	this->print();
}



void doMuld ( double* A, double* B, double* C, int BS)
{
        Muld(A,B,BS,BS,C,BS);
}



void Block::multiplyGPU(Block block1, Block block2)
{
	double *A, *B, *C;
        double **partitionsA, **partitionsB, **partitionsC;
        int nPartitions = 1;
        int M_part = M;
        int loop = true;
        while(loop){
                if ((M_part > MAX_BSIZE) && ((M%2)==0)){
                        M_part /= 2;
                        nPartitions *= 2;
                }
                else {
                        loop = false;
                }
        }


	partitionsA = (double **) malloc( nPartitions*nPartitions*sizeof( double *) );
        partitionsB = (double **) malloc( nPartitions*nPartitions*sizeof( double *) );
        partitionsC = (double **) malloc( nPartitions*nPartitions*sizeof( double *) );


        for (int i = 0; i < (nPartitions*nPartitions); i++){
                partitionsA[i] = (double*) malloc( M_part*M_part*sizeof( double) );
                partitionsB[i] = (double*) malloc( M_part*M_part*sizeof( double) );
                partitionsC[i] = (double*) malloc( M_part*M_part*sizeof( double) );
        }

        for (int i=0; i<M; i++){
                for (int j=0; j < M; j++){
                        int blockNum = (i/M_part)*nPartitions + j/M_part;
                        int row = i%M_part;
                        int col = j%M_part;
                        partitionsA[blockNum][row*M_part+col] = block1.data[i][j];
                        partitionsB[blockNum][row*M_part+col] = block2.data[i][j];
                        partitionsC[blockNum][row*M_part+col] = data[i][j];
                }
        }
	
	for (int i = 0; i < nPartitions; i++){
                for (int j = 0; j < nPartitions; j++){
                        for (int k = 0; k < nPartitions; k++){
                                doMuld(partitionsA[i*nPartitions+k], partitionsB[k*nPartitions+j], partitionsC[i*nPartitions+j], M_part);
                        }
                }
        }

        #pragma omp taskwait

	for (int i=0; i<M; i++)
        {
                for (int j=0; j<M; j++)
                {
                        int blockNum = (i/M_part)*nPartitions + j/M_part;
                        int row = i%M_part;
                        int col = j%M_part;
                        data[i][j] = partitionsC[blockNum][row*M_part+col];
                }
        }

}



#endif

void Block::print() {
	for (int i=0; i<M; i++) {
		for (int j=0; j<M; j++) {
			cout << data[i][j] << " ";
		}
		cout << "\r\n";
	}
}
