#include "Matmul.h"
#include "aux_func.h"
int N;  //MSIZE
int M;	//BSIZE
double **A, **B, **C;
double val;
void init_matrices(){
	A=(double**)malloc(N*N*sizeof(double*));
        B=(double**)malloc(N*N*sizeof(double*));
        C=(double**)malloc(N*N*sizeof(double*));
        for (int i = 0; i < N; i++){
        	for (int j = 0; j < N; j++){
                	A[i*N+j]=init_block(val, M*M);
                        B[i*N+j]=init_block(val, M*M);
                        C[i*N+j]=init_block(0.0, M*M); 
                }
        }
}
int main(int argc, char **argv) {
	N = atoi(argv[1]);
	M = atoi(argv[2]);
	val = atof(argv[3]);
	compss_on();
	init_matrices();
	for (int i=0; i<N; i++) {
               	for (int j=0; j<N; j++) {
                       	for (int k=0; k<N; k++) {
				multiply_blocks(A[i*N+k], B[k*N+j], C[i*N+j], M);
                       	}
               }
        }
	compss_barrier();
	compss_wait_on(C[0]);
	result(C[0]);
	compss_off();
	return 0;
}
