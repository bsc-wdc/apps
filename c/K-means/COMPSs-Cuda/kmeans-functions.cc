/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
// Copyright (c) 2017 Jorge Ejarque // COMPSs/OmpSs porting
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "kmeans.h"
#include "kernel.h"

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
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

/*----< find_nearest_cluster() >---------------------------------------------*/
/*#pragma omp task*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

void init_Fragment(Fragment *frag, char* filePath, int isBinaryFile){
        frag->init_file(filePath, isBinaryFile);
}


void update_clusters(Clusters *newClusters, ClustersSize *newClustersSize, Clusters *clusters){
	for (int i=0; i<(clusters->numClusters); i++){
            for (int j=0; j<(clusters->numCoords); j++) {
                if (newClustersSize->size[i] > 0){
                    clusters->coords[i][j] = newClusters->coords[i][j] / newClustersSize->size[i];
                }
            }
        }
}

void compute_newCluster(Fragment *frag, Clusters *clusters, Clusters *newClusters, ClustersSize *newClustersSize){
	int *index = new int[frag->numObjs]; 
	int i,j;
	clusters->print();
	newClusters->init(clusters->numClusters, clusters->numCoords);
    newClustersSize->init(clusters->numClusters);
	//printf("Computing cluster with %i element\n", frag.numObjs);

    for (i=0; i<frag->numObjs; i++) {
        /* find the array index of nestest cluster center */
		#pragma omp task
        index[i] = find_nearest_cluster(clusters->numClusters, frag->numCoords, frag->objects[i],clusters->coords);
	    //printf("[obj %i: %i]->",i, index);
            /* update new cluster centers : sum of objects located within */
	}
    #pragma omp taskwait
	for (i=0; i<frag->numObjs; i++) {
        newClustersSize->size[index[i]]++;
	    for (j=0; j<frag->numCoords; j++){
//            if (newClustersSize->size[index[i]]==1){
//			    newClusters->coords[index[i]][j]= 0.0;
//		    }
		    newClusters->coords[index[i]][j] += frag->objects[i][j];
	    }
    }

}


void do_cuda_find_nearest_cluster(int nCoord, int nObj, int nClust, float *deviceObjects, float *deviceClusters, int *membership, int numThreadsPerClusterBlock, int numClusterBlocks, int clusterBlockSharedDataSize){

	cout << "nCoord: " << nCoord << endl;
	cout << "nObj: " << nObj << endl;
	cout << "nClust: " << nClust << endl;

	cout << "device objects: " << endl;
	for (int i = 0; i < nCoord*nObj; i++) cout << deviceObjects[i] << endl;

	cout << "device clusters: " << endl;
        for (int i = 0; i < nCoord*nClust; i++) cout << deviceClusters[i] << endl;


//	cuda_find_nearest_cluster(nCoord, nObj, nClust,deviceObjects,deviceClusters,membership,numThreadsPerClusterBlock, numClusterBlocks, clusterBlockSharedDataSize);
}


//void cuda_compute_newCluster(Fragment *frag, Clusters *clusters, Clusters *newClusters, ClustersSize *newClustersSize, float *deviceObjects, float *deviceClusters, int numThreadsPerClusterBlock, int numClusterBlocks, int clusterBlockSharedDataSize){
void cuda_compute_newCluster(Fragment *frag, Clusters *clusters, Clusters *newClusters, ClustersSize *newClustersSize){//, int numThreadsPerClusterBlock, int numClusterBlocks, int clusterBlockSharedDataSize){
//	cout << "inside cuda function" << endl;
//   	int *index = new int[frag->numObjs];
	int index = 0;
        int i,j;
//        clusters->print();
//	cout << "startng init" << endl;
        newClusters->init(clusters->numClusters, clusters->numCoords);
//	cout << "numclusters is " << clusters->numClusters << endl;
    	newClustersSize->init(clusters->numClusters);
//	cout << "new clusters:" << endl;
//	clusters->print();
//	cout << "size[0] is " << newClustersSize->size[0] << endl;
//	cout << "ending init" << endl;
	//int membership[frag->numObjs];

	//cuda_find_nearest_cluster(frag->numCoords, frag->numObjs, clusters->numClusters, frag->objects, clusters, membership, numThreadsPerClusterBlock, numClusterBlocks, clusterBlockSharedDataSize);
	int a = 0; int b = 1; int c = 2; int d=3; int e=4; int f=5; 
	//cuda_find_nearest_cluster(a,b,c,d,e,f);

	int n = 3;
	int objects[n];
	int out[n];
	int *membership;
	int nObj = frag->numObjs;
	int nCoord = frag->numCoords;
	int nClust = clusters->numClusters;
	membership = (int*) malloc(nObj * sizeof(int));

/*
	float deviceObjects[nObj*nCoord];
	float deviceClusters[nClust*nCoord];

	for (i = 0; i < nCoord; i++){
            for (j = 0; j < nObj; j++){
//		cout << "adding element " << i*nObj+j << endl;
                deviceObjects[i*nObj+j] = frag->objects[j][i];
            }
        }

        for (i = 0; i < nCoord; i++){
            for (j = 0; j < nClust; j++){
//		cout << "copying element " << i*nClust+j << endl;
                deviceClusters[i*nClust+j] = deviceObjects[i*nObj+j];
            }
    	}
*/


	float **deviceObjects;
    	float **deviceClusters;

	deviceObjects = (float **)malloc(nCoord * sizeof(float *));
	assert(deviceObjects != NULL); 
	deviceObjects[0] = (float *)malloc(nCoord * nObj * sizeof(float));
	assert(deviceObjects[0] != NULL); 
	for (size_t i = 1; i < nCoord; i++) 
	    deviceObjects[i] = deviceObjects[i-1] + nObj;

        deviceClusters = (float **)malloc(nCoord * sizeof(float *));
        assert(deviceClusters != NULL);
        deviceClusters[0] = (float *)malloc(nCoord * nClust * sizeof(float));
        assert(deviceClusters[0] != NULL);
        for (size_t i = 1; i < nCoord; i++)
            deviceClusters[i] = deviceClusters[i-1] + nClust;


//  	malloc2D(deviceObjects, nCoord, nObj, float);
    	for (i = 0; i < nCoord; i++) {
            for (j = 0; j < nObj; j++) {
//          	deviceObjects[i*nCoord+j] = frag->objects[j][i];
		deviceObjects[i][j] = frag->objects[j][i];
            }
    	}

    	// pick first numClusters elements of objects[] as initial cluster centers
//  	malloc2D(deviceClusters, nCoord, nClust, float);
    	for (i = 0; i < nCoord; i++) {
            for (j = 0; j < nClust; j++) {
//                deviceClusters[i][j] = deviceObjects[i*nCoord+j];
//                deviceClusters[i][j] = deviceObjects[i][j];
		deviceClusters[i][j] = clusters->coords[j][i];
            }
    	}

    cout << "numCoords: " << frag->numCoords << endl;
        cout << "numObjs: " << frag->numObjs << endl;
        cout << "numClusters: " << clusters->numClusters << endl;

    cout << "objects:" << endl;

    for (int i = 0; i < frag->numObjs; i++){
        for (int j = 0; j < frag->numCoords; j++){
            cout << deviceObjects[j][i] << " " << endl;
        }
        cout << endl;
    }

    cout << "clusters:" << endl;

    for (int i = 0; i < clusters->numClusters; i++){
        for (int j = 0; j < frag->numCoords; j++){
            cout << deviceClusters[j][i] << " " << endl;
        }
        cout << endl;
    }


/*
	float* deviceObjects;
	float* deviceClusters;

	deviceObjects = (float *)malloc(nCoord * nObj * sizeof(float *));
        assert(deviceObjects != NULL); 

        deviceClusters = (float *)malloc(nCoord * nClust * sizeof(float));
        assert(deviceClusters != NULL);



        for (i = 0; i < nCoord; i++) {
            for (j = 0; j < nObj; j++) {
		cout << "copying object " << i << "," << j << " value " << frag->objects[j][i] << endl;
                deviceObjects[j*nCoord+i] = frag->objects[j][i];
//                deviceObjects[i][j] = frag->objects[j][i];
            }
        }

        for (i = 0; i < nCoord; i++) {
            for (j = 0; j < nClust; j++) {
                deviceClusters[j*nCoord+i] = deviceObjects[j*nCoord+i];
//                deviceClusters[i][j] = deviceObjects[i][j];
            }
        }

*/

	int numThreadsPerClusterBlock = 128;
	int numClusterBlocks = 2;
	int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);
//	cout << "CALLING CUDA" << endl;

		
//	for (int it = 0; it < frag->numObjs; it += 64){
//		cout << "launching iteration " << it << endl;
//		int task_size = min(64,frag->numObjs-it);
//		cout << "size is " << task_size << ", numObjs is " << frag->numObjs<< endl;

		//cout << "first elements: " << deviceObjects[it] << endl;
		//cout << deviceClusters[it][0] << endl;
//		cout << membership[it] << endl;

		cuda_find_nearest_cluster(frag->numCoords, frag->numObjs, clusters->numClusters, &deviceObjects[0][0],deviceClusters[0],&membership[0],numThreadsPerClusterBlock, numClusterBlocks, clusterBlockSharedDataSize);
//	}

//	cout << "cuda done" << endl;
	//cuda_find_nearest_cluster(a,b,c,objects,out,mem,d,e,f);
	//cuda_test(n, objects, out);

//	cout << "before taskwait" << endl;
	#pragma omp taskwait
//	cout << "after taskwait" << endl;

	for (i=0; i<frag->numObjs; i++) {
//	    cout << "inside object" << endl;
            index = membership[i];
	    cout << "index " << i << " is " << index << endl;
	    
            newClustersSize->size[index]++;

//	   cout << "entering frag" << endl;
            for (j=0; j<frag->numCoords; j++){
		//cout << "doing index " << j << endl;
//		cout << "accessing position " << index << " " << j << endl;
                newClusters->coords[index][j] += frag->objects[i][j];
	    }
        }

        free(membership);
}


void merge_newCluster(Clusters *newClusters_1, Clusters *newClusters_2, ClustersSize *newClustersSize_1, ClustersSize *newClustersSize_2){
        int i, j;
		
        for (i=0; i<newClusters_1->numClusters; i++){
                newClustersSize_1->size[i] = newClustersSize_1->size[i] + newClustersSize_2->size[i];
			#pragma omp task
            for (j=0; j<newClusters_1->numCoords; j++) {
                newClusters_1->coords[i][j] = newClusters_1->coords[i][j] + newClusters_2->coords[i][j];
            }
        }
		#pragma omp taskwait
}

