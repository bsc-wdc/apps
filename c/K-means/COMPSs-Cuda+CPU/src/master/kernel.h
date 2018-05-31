#include <stdio.h>


#pragma omp target device(cuda) copy_deps ndrange(1, 1, 1)
#pragma omp task in(objects[0:1], deviceClusters[0:1])  task out(membership[0:1])
__global__ void cuda_find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *objects, float *deviceClusters, int *membership, int numThreadsPerClusterBlock, int numClusterBlocks, int clusterBlockSharedDataSize);

//#pragma omp target device(cuda) copy_deps ndrange(1, 1, 1)
//#pragma omp task in(objects[0:numObjs-1], deviceClusters[0:numClusters])  task out(membership[0:numObjs-1])
//__global__  void cuda_find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *objects, float *deviceClusters, int *membership, int numThreadsPerClusterBlock, int numClusterBlocks, int clusterBlockSharedDataSize);


#pragma omp target device(cuda) copy_deps ndrange(1, 1, 1)
#pragma omp task in(objects[0:n-1])  task out(out[0:n-1])
__global__ void cuda_test(int n, int *objects, int *out);
