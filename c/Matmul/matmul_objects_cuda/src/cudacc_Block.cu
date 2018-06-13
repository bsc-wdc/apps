void Muld(double *A, double *B, int wA, int wB, double *C, int NB) __attribute__((global));
void gpu_ol_Muld_2_1880598850_unpacked(double *A, double *B, int wA, int wB, double *C, int NB) throw()
{
  {
    struct dim3 dimGrid /* ((1), (1), (1)) */ ;
    struct dim3 dimBlock /* ((1), (1), (1)) */ ;
    dimBlock.x = 64 < 32 ? 64 : 32;
    dimGrid.x = 64 < 32 ? 1 : 64 / 32 + (64 % 32 == 0 ? 0 : 1);
    dimBlock.y = 64 < 32 ? 64 : 32;
    dimGrid.y = 64 < 32 ? 1 : 64 / 32 + (64 % 32 == 0 ? 0 : 1);
    dimBlock.z = 1;
    dimGrid.z = 1;
    ::Muld<<<dimGrid, dimBlock, 0, ::nanos_get_kernel_execution_stream()>>>(A, B, wA, wB, C, NB);
  }
}
void gpu_ol_Muld_4_1880598850_unpacked(double *A, double *B, int wA, int wB, double *C, int NB) throw()
{
  {
    struct dim3 dimGrid /* ((1), (1), (1)) */ ;
    struct dim3 dimBlock /* ((1), (1), (1)) */ ;
    dimBlock.x = 64 < 32 ? 64 : 32;
    dimGrid.x = 64 < 32 ? 1 : 64 / 32 + (64 % 32 == 0 ? 0 : 1);
    dimBlock.y = 64 < 32 ? 64 : 32;
    dimGrid.y = 64 < 32 ? 1 : 64 / 32 + (64 % 32 == 0 ? 0 : 1);
    dimBlock.z = 1;
    dimGrid.z = 1;
    ::Muld<<<dimGrid, dimBlock, 0, ::nanos_get_kernel_execution_stream()>>>(A, B, wA, wB, C, NB);
  }
}
