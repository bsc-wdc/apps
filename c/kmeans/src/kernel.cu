#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
static inline int nextPowerOfTwo(int n) {
    n--;
    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints
    return ++n;
}
 __host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{   
    int i;
    float ans=0.0;
    for (i = 0; i < numCoords; i++) { 
        ans += (objects[(objectId*numCoords) + i] - clusters[(clusterId*numCoords) +i]) *
               (objects[(objectId*numCoords) + i] - clusters[(clusterId*numCoords) +i]);
    }
    return(ans);
}
__global__ 
void cuda_find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
			  float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership
			)
{
    float *clusters = deviceClusters;
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    //if (objectId == 1) printf("A cuda call with numObjs: %d, numCoords: %d, numClusters:%d \n", numObjs, numCoords, numClusters);    
    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);
        /*if (objectId == 1) {
		printf("Distance to cluster 0: %f\n", min_dist);

        }*/
	__syncthreads();
        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /*if (objectId == 1) {
		printf("Distance to cluster 0: %f\n", min_dist);
	    }*/
            if (dist < min_dist) { // find the min and its array index 
                min_dist = dist;
                index    = i;
            }
        }
	__syncthreads();
        /*if (objectId == 1) {
                printf("Object assigned to cluster %d\n", index);
        }*/

        membership[objectId] = index;
    }
}
