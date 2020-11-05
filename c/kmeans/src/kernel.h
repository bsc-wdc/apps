#include <stdio.h>

#ifdef OMPSS2_ENABLED
#pragma oss task in([numObjs*numCoords]deviceObjects, [numClusters*numCoords]deviceClusters) out([numObjs]membership) device(cuda) ndrange(1, numObjs, 64) label("findNearestClusterGPU")
__global__ void cuda_find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *deviceObjects, float *deviceClusters, int *membership);
#endif

#ifdef OMPSS_ENABLED
#pragma omp target device(cuda) copy_deps ndrange(1, numObjs, 64)
#pragma omp task in(deviceObjects[0;(numObjs*numCoords)], deviceClusters[0;(numClusters*numCoords)]) out(membership[0;(numObjs)])
__global__ void cuda_find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *deviceObjects, float *deviceClusters, int *membership);
#endif

