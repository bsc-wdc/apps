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

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
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
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }
     
    return(ans);
}


/*----< cuda_find_nearest_cluster() >---------------------------------------------*/

__global__ 
void cuda_find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
			  float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership
			)
{
  

//    __shared__ int sharedMemory[8192];

//    unsigned char *membershipChanged = (unsigned char *)sharedMemory;

    float *clusters = deviceClusters;
    
//    membershipChanged[threadIdx.x] = 0;

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

//    printf("blockDim.x: %d\n", blockDim.x);
//    printf("blockIdx.x: %d\n", blockIdx.x);
//    printf("threadIdx.x: %d\n", threadIdx.x);

//    if (objectId == 0) printf("this is a cuda call\n");

    if (objectId < numObjs) {


        int   index, i;
        float dist, min_dist;

	// find the cluster id that has min distance to object 
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);
	__syncthreads();
	
        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            if (objectId == 0) {
//                printf("object %d: dist %f\n", objectId, dist);
	    }
            // no need square root 
            if (dist < min_dist) { // find the min and its array index 
                min_dist = dist;
                index    = i;
//                if (objectId == 0) printf("object %d: index %d, i %d, dist %f, min_dist %f\n", objectId, index, i, dist, min_dist);
            }
        }
	__syncthreads();



        // assign the membership to object objectId 
        membership[objectId] = index;

//        printf("object %d: index %d, min_dist %f, member %d\n", objectId, index, min_dist, membership[objectId]);

//        __syncthreads();    //  For membershipChanged[]

    }

//    printf("cuda task ended\n");

}
