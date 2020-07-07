#define BLOCK_SIZE 32
#pragma omp target device(opencl) copy_deps ndrange(2, 32, 32, 8, 8) file(kernel.cl)
#pragma omp task in(A[0:wA*wA], B[0:wB*wB]) inout(C[0:wA*wB]) 
__kernel void Muld(__global double* A, __global double* B, int wA, int wB, __global double* C);
