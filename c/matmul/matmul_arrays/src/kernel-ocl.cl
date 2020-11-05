#include "kernel.h"
__kernel void Muld(__global double* A,__global double* B, int wA, int wB,__global double* C) {
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1);
  double Csub = 0;
  int used=0;
  for (int a = wA * BLOCK_SIZE * by, b = BLOCK_SIZE * bx; a <= Begin + wA - 1; a += BLOCK_SIZE, b += BLOCK_SIZE * wB) {
	used=1;
    __local double As[BLOCK_SIZE][BLOCK_SIZE],  Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[ty][tx] = A[a + wA * ty + tx];  
    Bs[ty][tx] = B[b + wB * ty + tx];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < BLOCK_SIZE; ++k)
      Csub += As[ty][k] * Bs[k][tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] += Csub;
}
