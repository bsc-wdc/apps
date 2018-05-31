#pragma omp target device(opencl) copy_deps ndrange( 1,n,128 ) file(fpga_kernel/saxpy_kernel.aocx)
#pragma omp task in(([n]x)[0;n]) inout(([n]y)[0;n])
__kernel void saxpy_fpga(int n, float a,
       __global float* x, __global float* y);
