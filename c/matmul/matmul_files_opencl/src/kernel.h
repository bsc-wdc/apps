
// Thread block size
#define BLOCK_SIZE 32


#ifdef __cplusplus
extern "C"
{
#endif


//Kernel declaration as a task should be here
//Remember, we want to multiply two matrices, (A*B=C) where all of them have size NB*NB
#pragma omp target device(opencl) copy_deps ndrange(2, 32, 32, 8, 8) file(kernel.cl)
#pragma omp task in(A[0:NB*NB], B[0:NB*NB]) inout(C[0:NB*NB]) 
__kernel void Muld(__global double* A, __global double* B, int wA, int wB, __global double* C, int NB);


#ifdef __cplusplus
}
#endif

