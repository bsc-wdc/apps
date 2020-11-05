#include <stdio.h>

// Thread block size
#define BLOCK_SIZE 32


//#ifdef __cplusplus
//extern "C"
//{
//#endif


//Kernel declaration as a task should be here
//Remember, we want to multiply two matrices, (A*B=C) where all of them have size NB*NB
#pragma omp target device(cuda) copy_deps ndrange(2, 64, 64, 32, 32)
#pragma omp task in(A[0:NB*NB], B[0:NB*NB]) inout(C[0:NB*NB])
__global__ void Muld(double* A, double* B, int wA, int wB, double* C, int NB);


//#ifdef __cplusplus
//}
//#endif

