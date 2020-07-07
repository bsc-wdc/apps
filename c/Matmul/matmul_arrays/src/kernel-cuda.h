#include <stdio.h>
#define BLOCK_SIZE 32
#ifdef OMPSS_ENABLED
#pragma omp target device(cuda) copy_deps ndrange(2, 64, 64, 32, 32)
#pragma omp task in(A[0:wA*WA], B[0:wB*wB]) inout(C[0:WB*WA])
__global__ void Muld(double* A, double* B, int wA, int wB, double* C);
#endif

#ifdef OMPSS2_ENABLED
#pragma oss task in(A[0:wA*WA], B[0:wB*wB]) inout(C[0:WB*WA]) device(cuda) ndrange(2, 64, 64, 32, 32)
__global__ void Muld(double* A, double* B, int wA, int wB, double* C);
#endif

