#include <stdio.h>

#pragma omp target device(cuda) copy_deps ndrange(1, numObjs, 1024)
#pragma omp task in(deviceObjects[0:(numObjs*numCoords)], deviceClusters[0:(numClusters*numCoords)]) out(membership[0:(numObjs)])
__global__ void cuda_find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *deviceObjects, float *deviceClusters, int *membership);

