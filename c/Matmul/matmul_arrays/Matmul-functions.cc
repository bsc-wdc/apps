#include "Matmul.h"

#ifdef CUDA_ENABLED
#include "kernel-cuda.h"
#endif
#ifdef OCL_ENABLED
#include "kernel-ocl.h"
#endif


double *init_block(double value, int size){
     double *block = (double*) malloc(size*sizeof(double));
     for (int i = 0; i < size; i++){
          block[i]=value;
     }
     return block;
}

void multiply_blocks(double *blockA, double *blockB, double *blockC, int M) {

        for (int i=0; i<M; i++) {
#ifdef OMPSS2_ENABLED
#pragma oss task firstprivate(i) in([M*M]blockA, [M*M]blockB) out(blockC[i*M;M])
#endif
#ifdef OMPSS_ENABLED
#pragma omp task firstprivate(i) in(blockA[0;M*M], blockB[0;M*M]) out(BlockC[i*M;M])
#endif
		for (int j=0; j<M; j++) {
                        for (int k=0; k<M; k++) {
                                blockC[i*M+j] += blockA[i*M+k] * blockB[k*M+j];
                        }
                }
        }
#ifdef OMPSS2_ENABLED
	#pragma oss taskwait
#endif
#ifdef OMPSS_ENABLED
	#pragma omp taskwait
#endif

}

void multiply_blocks_GPU(double *blockA, double *blockB, double *blockC, int M) {
#ifdef CUDA_ENABLED
      Muld(blockA,blockB,M,M,blockC);
#endif
#ifdef OCL_ENABLED
      Muld(blockA,blockB,M,M,blockC);
#endif
#ifdef OMPSS2_ENABLED
        #pragma oss taskwait in([M*M]blockC)
#endif
#ifdef OMPSS_ENABLED
        #pragma omp taskwait in(blockC[0;M*M])
#endif
