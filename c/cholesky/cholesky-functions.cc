#include "cholesky.h"

#include <stdlib.h>
#include <time.h>

void identity_multiplied(int N, int factor, double* A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j)
                A[i*N + i] = factor + A[i*N + i];
        }
    }
}

void generate_block(double* A, int N, int D) {

    /*
        * A es un bloque de size N

        * N es el size del bloque

        * D indica si el bloque contiene parte
            de la diagonal de la matriz total.
    */

    //Initialize a block to random numbers.
    for(int i = 0; i < N*N; i++) 
        A[i]=((float)rand())/((float)RAND_MAX);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[j*N + i] = A[j*N + i] + A[i*N + j];
            A[i*N + j] = A[j*N + i];
        }
    }

    if (D) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == j)
                    A[i*N + i] = 2*N + A[i*N + i];
            }
        }
    }

    return;
}

/*
int ddss_dpotrf(enum DDSS_UPLO UPLO, int N, double* A, int LDA) {
    return 1;
}

int ddss_dtrsm( enum DDSS_SIDE SIDE, enum DDSS_UPLO UPLO, 
		             enum DDSS_TRANS TRANS_A, enum DDSS_DIAG DIAG, 
            		 int M, int N,
                     double ALPHA, double* A, int LDA,
                     double* B, int LDB) {
    return 1;
}
    
int ddss_dgemm( enum DDSS_TRANS TRANS_A, enum DDSS_TRANS TRANS_B,
	                 int M, int N, int K,
                     double ALPHA, double* A, int LDA,
                     double* B, int LDB,
                     double BETA,  double* C, int LDC ) {
    return 1;
}

    
int ddss_dsyrk( enum DDSS_UPLO UPLO, enum DDSS_TRANS TRANS,
                     int N, int K,
                     double ALPHA, double* A, int LDA,
                     double BETA,  double* C, int LDC ) {
    return 1;
}
*/
