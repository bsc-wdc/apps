#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "kmeans.h"
#include "kmeans_io.h"

#ifdef CUDA_ENABLED
#include "kernel.h"

void compute_newCluster_GPU(int numObjs, int numCoords, int numClusters, int sizeFrags, int sizeClusters, int* index_frag, float *frag, float *clusters, float *newClusters, int *newClustersSize){
        int *index = new int[numObjs];

        cuda_find_nearest_cluster(numCoords, numObjs, numClusters, frag, clusters, index);

        for (int i=0; i<numClusters; i++) {
            newClustersSize[i]=0;
        }

#ifdef OMPSS2_ENABLED
        #pragma oss taskwait in([numObjs]index)
#endif
#ifdef OMPSS_ENABLED
        #pragma omp taskwait in(index[0;numObjs])
#endif
        for (int i=0; i<numObjs; i++) {
            newClustersSize[index[i]]++;
            for (int j=0; j<numCoords; j++){
                if (newClustersSize[index[i]]==1){
                        newClusters[index[i]*numCoords+j]= 0.0;
                }
                newClusters[index[i]*numCoords+j] += frag[i*numCoords + j];
            }
        }
}
#endif

__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;
    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
    return(ans);
}

__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float *clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, &clusters[0]);
    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

float *init_Fragment(int numObjs, int numCoords, int sizeFrags, char* filePath, int* frag_index){
   float *frag = file_read(false, filePath, numObjs, numCoords);
   return frag;	
}

void compute_newCluster(int numObjs, int numCoords, int numClusters, int sizeFrags, int sizeClusters, int* frag_index, float *frag, float *clusters, float *newClusters, int *newClustersSize){
    int block=100;
    int *index = (int*)malloc(numObjs * sizeof(int)); 

//    printf("Number of numObjs: %d\n", numObjs);

    int k, i, j;
    for (k=0; k<numObjs; k+=block) {
#ifdef OMPSS2_ENABLED
#pragma oss task firstprivate(k, i) shared(index) in([numCoords*numObjs]frag, [numClusters*numCoords]clusters)
#endif
#ifdef OMPSS_ENABLED
#pragma omp task firstprivate(k) in(frag[0;numCoords*numObjs], clusters[0;numClusters*numCoords]) concurrent(index[0;numObjs])
#endif
{
        for (i=k; (i<numObjs) && (i<(k+block)); i++) {
            index[i] = find_nearest_cluster(numClusters, numCoords, &frag[i*numCoords], clusters);
        }
}
    }

    for (i=0; i<numClusters; i++) {
        newClustersSize[i]=0;
    }

#ifdef OMPSS2_ENABLED
#pragma oss taskwait
#endif
#ifdef OMPSS_ENABLED
#pragma omp taskwait in(index[0;numObjs])
#endif

    for (i=0; i<numObjs; i++) {
        newClustersSize[index[i]]++;
#ifdef OMPSS2_ENABLED
        #pragma oss task concurrent([numClusters*numCoords]newClusters)
#endif
        for (j=0; j<numCoords; j++){
            if (newClustersSize[index[i]]==1){
                newClusters[index[i]*numCoords+j]= 0.0;
            }
            newClusters[index[i]*numCoords+j] += frag[i*numCoords + j];
        }
    }
}

void merge_newCluster(int numCoords, int numClusters, int sizeClusters, float *newClusters_1, float *newClusters_2, int *newClustersSize_1, int *newClustersSize_2){
        int i, j;
        
	for (i=0; i<numClusters; i++){
            newClustersSize_1[i] = newClustersSize_1[i] + newClustersSize_2[i];
#ifdef OMPSS2_ENABLED
            #pragma oss task concurrent([numClusters*numCoords]newClusters_1)
#endif
            for (j=0; j<numCoords; j++) {
                newClusters_1[i*numCoords+j] += newClusters_2[i*numCoords+j];
            }
        }
}

void update_Clusters(int numCoords, int numClusters, int sizeClusters, float *clusters, float *newClusters, int *newClustersSize){
        int i, j;
    	for (i=0; i<numClusters; i++){
#ifdef OMPSS2_ENABLED
            #pragma oss task concurrent([numClusters*numCoords]clusters)
#endif
            for (j=0; j<numCoords; j++) {
                if (newClustersSize[i] > 0){
                    clusters[i*numCoords+j] = newClusters[i*numCoords+j] / newClustersSize[i];
                }
            }
        }
}

