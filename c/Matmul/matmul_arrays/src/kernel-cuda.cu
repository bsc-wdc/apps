#include "kernel.h"
__global__ void Muld(double* A, double* B, int wA, int wB, double* C){
   int aEnd   = wA * BLOCK_SIZE * blockIdx.y + wA - 1;
   double Csub = 0;
   for (int a = wA * BLOCK_SIZE * blockIdx.y, b = BLOCK_SIZE * blockIdx.x; a <= aEnd; a += BLOCK_SIZE, b += BLOCK_SIZE * wB) {
      __shared__ double As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];
      As[threadIdx.y][threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];  
      Bs[threadIdx.y][threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];
      __syncthreads();
      for (int k = 0; k < BLOCK_SIZE; ++k)
         Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
      __syncthreads();
   }
   C[wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x + wB*threadIdx.y + threadIdx.x] += Csub;
}




