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
#include "Matrix.h"

Matrix::Matrix(int mSize) {
	N = mSize;
	data.resize(N);
	for (int i=0; i<N; i++) {
		data[i].resize(N);
	}
}

Matrix Matrix::init(int mSize, int bSize, double val) {
	Matrix matrix(mSize);
	for (int i=0; i<mSize; i++) {
		for (int j=0; j<mSize; j++) {
			matrix.data[i][j] = Block::init(bSize, val);
		}
	}
	return matrix;
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

