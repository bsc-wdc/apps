#include <stdio.h>

#pragma omp target device(cuda) copy_deps ndrange(1, numObjs, 64)
#pragma omp task in(deviceObjects[0:(numObjs*numCoords-1)], deviceClusters[0:(numClusters*numCoords-1)]) out(membership[0:(numObjs-1)])
__global__ void cuda_find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *deviceObjects, float *deviceClusters, int *membership, int numThreadsPerClusterBlock, int numClusterBlocks, int clusterBlockSharedDataSize);
