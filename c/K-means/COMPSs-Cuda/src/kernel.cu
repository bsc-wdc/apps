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

    if(objectId == 0){
    	printf("coords used in first element\n");
    	for (i = 0; i < numCoords; i++){
		printf("%f\n", objects[numObjs * i + objectId]);
    	}
	printf("cluster coords\n");
	for (i = 0; i < numCoords; i++){
                printf("%f\n", clusters[numClusters * i + clusterId]);
        }
    }

                                                                                                     
    for (i = 0; i < numCoords; i++) { 
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }
     
    if(objectId == 0){
	printf("first ans is %f for cluster %d\n", ans, clusterId);
    }
                                                                                                                                                                        
    return(ans);
}

/*__global__ void cuda_test(int n, int *objects, int *out){

}
*/

/*
__global__
void cuda_find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,
                          int numThreadsPerClusterBlock,
                          int numClusterBlocks,
                          int clusterBlockSharedDataSize)
{*/






/*----< cuda_find_nearest_cluster() >---------------------------------------------*/

__global__ 
void cuda_find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
			  float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,
			  int numThreadsPerClusterBlock, 
			  int numClusterBlocks, 
			  int clusterBlockSharedDataSize
			)
{
  
//   printf("hello from cuda task \n"); 



   //extern __shared__ char sharedMemory[];

    __shared__ char sharedMemory[128];

    unsigned char *membershipChanged = (unsigned char *)sharedMemory;

//#if BLOCK_SHARED_MEM_OPTIMIZATION
//    float *clusters = (float *)(sharedMemory + blockDim.x);
//#else
    float *clusters = deviceClusters;
//#endif
    
    membershipChanged[threadIdx.x] = 0;

//#if BLOCK_SHARED_MEM_OPTIMIZATION
    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates! For reference, a Tesla C1060 has 16
    //  KiB of shared memory per block, and a GeForce GTX 480 has 48 KiB of
    //  shared memory per block.
//    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
//        for (int j = 0; j < numCoords; j++) {
//    	    clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
//        }
//    }
//    __syncthreads();
//#endif
    
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;


    if (objectId < numObjs) {
//	printf("task %d reporting, nObjects is %d, values are %d \n %d \n %d \n %d \n %d \n %d \n %d \n %d \n %d \n %d \n\n", objectId, numObjs, objects[objectId], objects[numObjs * 1 + objectId], objects[numObjs * 2 + objectId], objects[numObjs * 3 + objectId], objects[numObjs * 4 + objectId], objects[numObjs * 5 + objectId], objects[numObjs * 6 + objectId], objects[numObjs * 7 + objectId], objects[numObjs * 8 + objectId], objects[numObjs * 9 + objectId]);


/*
	if (objectId == 0){
		printf("objects:\n");
		for (int i = 0; i < numObjs*numCoords; i++){
			printf("%f\n", objects[i]);
		}
	}
*/




//	printf("task %d reporting, membership values are %d \n %d \n %d \n %d \n %d \n %d \n %d \n %d \n %d \n %d \n\n", objectId, membership[0], membership[1], membership[2], membership[3], membership[4], membership[5], membership[6], membership[7], membership[8], membership[9]);	

        int   index, i;
        float dist, min_dist;

	// find the cluster id that has min distance to object 
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);
	__syncthreads();
	printf("distance 0 for object is %f\n", objectId, min_dist);
	
        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            // no need square root 
		printf("distance 1 for object %d is %f\n", objectId, dist);
            if (dist < min_dist) { // find the min and its array index 
                min_dist = dist;
                index    = i;
            }
        }
	__syncthreads();
	printf("chosen dist for object %d is %f\n", objectId, min_dist);


	if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

//	printf("object id %d has membership %d\n", objectId, index);	

        // assign the membership to object objectId 
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

	printf("object id %d has membership %d\n", objectId, index);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

//	membership[objectId] = 123;

    }

/*
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;


    if (objectId < numObjs) {
	membership[objectId] = 123;
    }
*/

	printf("end of cuda task \n");

}
