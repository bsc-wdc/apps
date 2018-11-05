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

#include "Block.h"
#include <sys/types.h>
#include <sys/syscall.h>
#include <iostream>
#define gettid() syscall(SYS_gettid)

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

void Block::multiply(Block block1, Block block2) {

	for (int i=0; i<M; i++) {
        #pragma omp task shared(block1, block2, data)
		for (int j=0; j<M; j++) {
			for (int k=0; k<M; k++) {
				data[i][j] += block1.data[i][k] * block2.data[k][j];
			}
		}
	}
	#pragma omp taskwait
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
