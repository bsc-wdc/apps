#include <iostream>
#include "aux_func.h"

void print_block(int M, double *block) {
        for (int i=0; i<M; i++) {
                for (int j=0; j<M; j++) {
                        cout << block[(i*M)+j] << " ";
                }
                cout << "\r\n";
        }
}

void result(double *block) {
        cout << block[0] << "\r\n";
}

